#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import urllib.request
import shutil
import sys
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import librosa
from tqdm import tqdm
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel

# VGGish PyTorch
from torchvggish import vggish, vggish_input

# URL for BPE (raw gz) from the official CLIP repo
BPE_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
BPE_LOCAL_REL = "utils/bpe_simple_vocab_16e6.txt.gz"

def ensure_bpe(repo_root: Path):
    """
    Ensure that utils/bpe_simple_vocab_16e6.txt.gz exists and is a valid gzip file.
    If not, try to download it from the CLIP repository.
    """
    bpe_path = repo_root / BPE_LOCAL_REL
    bpe_path.parent.mkdir(parents=True, exist_ok=True)
    def is_gzip(path: Path):
        try:
            with open(path, "rb") as f:
                magic = f.read(2)
            return magic == b'\x1f\x8b'
        except Exception:
            return False

    if bpe_path.exists() and is_gzip(bpe_path):
        print(f"[ok] BPE file exists and looks gzipped: {bpe_path}")
        return str(bpe_path)

    print(f"[info] BPE file missing or not gzipped at {bpe_path}. Downloading from {BPE_URL} ...")
    tmp_path = repo_root / (bpe_path.name + ".tmp")
    try:
        urllib.request.urlretrieve(BPE_URL, tmp_path)
        # verify magic
        if not is_gzip(tmp_path):
            # try follow redirects via curl fallback
            print("[warn] downloaded file is not gz; trying curl -L fallback ...")
            try:
                import subprocess
                subprocess.check_call(["curl", "-L", "-o", str(tmp_path), BPE_URL])
            except Exception:
                pass
        if not is_gzip(tmp_path):
            raise RuntimeError(f"Downloaded BPE is not gzipped. Check network or download manually to {bpe_path}")
        shutil.move(str(tmp_path), str(bpe_path))
        print(f"[ok] BPE downloaded to {bpe_path}")
        return str(bpe_path)
    except Exception as ex:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download BPE: {ex}")



def validate_audio_file(audio_path, sr=44100):
    """
    Validate an audio file with librosa (fast, basic).
    Returns (is_valid, error_message)
    """
    try:
        # Try to load the audio
        y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
        
        # Check that the audio is not empty
        if len(y) == 0:
            return False, "Empty audio (0 samples)"
        
        # Check sample rate
        if sr_loaded <= 0:
            return False, f"Invalid sample rate: {sr_loaded}"
        
        return True, None
    except librosa.LibrosaError as e:
        return False, f"LibrosaError: {str(e)[:50]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:50]}"

def validate_vggish_audio_file(audio_path):
    """
    Validate an audio file specifically for VGGish (wavfile_to_examples).
    Returns (is_valid, error_message)
    """
    try:
        examples = vggish_input.wavfile_to_examples(str(audio_path))
        
        if len(examples) == 0:
            return False, "VGGish generated no patches"
        
        return True, None
    except TypeError as e:
        return False, f"Invalid WAV format: {str(e)[:40]}"
    except Exception as e:
        return False, f"VGGish error: {str(e)[:50]}"

def validate_all_audio_files(audio_files, validation_mode='quick'):
    """
    Validate ALL audio files and return:
    - valid_stems: list of valid stems
    - invalid_stems: dict {stem: error_reason}
    
    validation_mode:
      'quick': use only librosa.load (fast)
      'vggish': use vggish_input.wavfile_to_examples (stricter, but slow)
    """
    print("\n" + "="*80)
    print("PRELIMINARY AUDIO FILE VALIDATION")
    print("="*80)
    print(f"Checking {len(audio_files)} audio files ({validation_mode} mode)...")
    
    valid_stems = []
    invalid_stems = {}
    
    for idx, audio_path in enumerate(tqdm(audio_files, desc="Audio validation"), start=1):
        stem = Path(audio_path).stem
        
        # Specific skip for problematic file at position 5652
        if idx == 5652:
            print(f"\nSkipping file at position {idx}: {stem} (problematic file)")
            invalid_stems[stem] = "Manually skipped (position 5652)"
            continue
        
        if validation_mode == 'vggish':
            is_valid, error = validate_vggish_audio_file(audio_path)
        else:
            is_valid, error = validate_audio_file(audio_path)
        
        if is_valid:
            valid_stems.append(stem)
        else:
            invalid_stems[stem] = error
    
    print(f"\nValidation completed:")
    print(f"  - Valid files: {len(valid_stems)}")
    print(f"  - Corrupted files: {len(invalid_stems)}")
    
    if invalid_stems:
        print(f"\nCorrupted audio files found:")
        for stem, error in list(invalid_stems.items())[:10]:
            print(f"  - {stem}: {error}")
        if len(invalid_stems) > 10:
            print(f"  ... and {len(invalid_stems) - 10} more corrupted files")
        
        # Save log of corrupted files
        log_path = Path("corrupted_audios.log")
        with open(log_path, "w", encoding='utf-8') as f:
            f.write(f"Audio validation: {len(invalid_stems)} corrupted files\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for stem, error in sorted(invalid_stems.items()):
                f.write(f"{stem}: {error}\n")
        print(f"\nLog saved: {log_path}")
    else:
        print(f"\nAll audio files are valid!")
    
    print("="*80)
    
    return valid_stems, invalid_stems

def summarize_failures(label: str, failures: Dict[str, str], max_items: int = 5) -> None:
    """Log a readable summary of skipped files."""
    if not failures:
        return
    print(f"[warn] {len(failures)} {label} skipped due to extraction errors.")
    for name, reason in list(failures.items())[:max_items]:
        print(f"  - {name}: {reason}")
    if len(failures) > max_items:
        print(f"  ... altri {len(failures) - max_items} non mostrati.")

def get_image_preprocess(model):
    try:
        res = model.visual.input_resolution.item()
    except Exception:
        res = getattr(model, "image_resolution", 224)
    preprocess = transforms.Compose([
        transforms.Resize(int(res * 256 / 224), interpolation=Image.BICUBIC),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    return preprocess, res

def load_model(weights, device):
    from model.audioclip import AudioCLIP
    if weights is None:
        model = AudioCLIP(pretrained=True)
    else:
        model = AudioCLIP(pretrained=weights)
    model.eval()
    model.to(device)
    return model

def load_vggish_model(device='cpu'):
    """
    Load the VGGish model using torchvggish (PyTorch).
    Returns the PyTorch model ready for inference.
    """
    print("[VGGish] Loading torchvggish model (PyTorch)...")
    model = vggish()
    model.eval()
    # Use specified device (GPU if available)
    model.to(device)
    print(f"[VGGish] Model loaded successfully on {device}")
    return model

def process_images_ordered(model, preprocess, image_items, device, batch_size=8):
    feats_dict: Dict[str, np.ndarray] = {}
    failures: Dict[str, str] = {}
    batch_imgs: List[torch.Tensor] = []
    batch_names: List[str] = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        img_tensor = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            feats = model.encode_image(img_tensor)  # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype(np.float32)
        for idx, name in enumerate(batch_names):
            feats_dict[name] = feats[idx]
        batch_imgs.clear()
        batch_names.clear()

    for name, path in image_items:
        try:
            with Image.open(path) as img:
                batch_imgs.append(preprocess(img.convert('RGB')))
            batch_names.append(name)
        except Exception as ex:
            failures[name] = f"{type(ex).__name__}: {ex}"
            continue
        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()
    return feats_dict, failures

def fix_audio_length(y, target_len):
    if len(y) >= target_len:
        return y[:target_len]
    else:
        pad = target_len - len(y)
        return np.pad(y, (0, pad))

def encode_audio_windows(model, aud_windows_tensor, device, batch_size=64):
    # aud_windows_tensor shape: (W, 1, samples)
    W = aud_windows_tensor.shape[0]
    feats = []
    aud_windows_tensor = aud_windows_tensor.to(device)
    with torch.no_grad():
        for i in range(0, W, batch_size):
            b = aud_windows_tensor[i:i+batch_size]
            f = model.encode_audio(b)  # (bsize, D)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    if len(feats) == 0:
        return np.zeros((0, model.embed_dim), dtype=np.float32)
    return np.vstack(feats)

def process_audios_sliding_mean(model, audio_items, device, sr=44100,
                                window_sec=2.0, stride_sec=1.0, window_batch_size=64):
    feats_dict: Dict[str, np.ndarray] = {}
    failures: Dict[str, str] = {}
    win = int(window_sec * sr)
    hop = int(stride_sec * sr)
    if win <= 0:
        raise ValueError("window_sec must be > 0")
    iterator = tqdm(audio_items, desc="audio files") if audio_items else []
    for name, path in iterator:
        try:
            y, _ = librosa.load(path, sr=sr, mono=True)
            if len(y) == 0:
                raise ValueError("empty audio")
            if len(y) < win:
                y_pad = fix_audio_length(y, win)
                windows = [y_pad]
            else:
                windows = []
                for start in range(0, len(y) - win + 1, hop):
                    windows.append(y[start:start+win])
                last_start = len(y) - win
                if (len(y) - win) % hop != 0 and last_start >= 0:
                    windows.append(y[last_start:last_start+win])
                if len(windows) == 0:
                    windows = [fix_audio_length(y, win)]
            wnd_arr = np.stack(windows, axis=0).astype(np.float32)
            wnd_t = torch.from_numpy(wnd_arr).unsqueeze(1)
            feats_w = encode_audio_windows(model, wnd_t, device, batch_size=window_batch_size)
            agg = feats_w.mean(axis=0) if feats_w.shape[0] > 0 else np.zeros((model.embed_dim,), dtype=np.float32)
            feats_dict[name] = agg.astype(np.float32)
        except Exception as ex:
            failures[name] = f"{type(ex).__name__}: {ex}"
    return feats_dict, failures

def process_texts_ordered(model, text_items, device, batch_size=32, max_tokens=77):
    """
    Process texts with sliding window if necessary.
    If a text exceeds max_tokens after tokenization, split it into chunks and average the embeddings.
    """
    feats_dict: Dict[str, np.ndarray] = {}
    failures: Dict[str, str] = {}
    
    # Import tokenizer from the model
    from model.clip.clip import tokenize
    
    for name, path in text_items:
        try:
            txt = Path(path).read_text(encoding='utf-8').strip()
            
            # Tokenize the text to see how long it is
            try:
                # Try to tokenize normally first
                with torch.no_grad():
                    feats = model.encode_text([[txt]])
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    feats_dict[name] = feats.cpu().numpy().astype(np.float32)[0]
            except RuntimeError as e:
                if "too long for context length" not in str(e):
                    raise
                
                # The text is too long, split it into sentences
                # Split by periods, keep separators
                sentences = []
                for part in txt.replace('! ', '.|').replace('? ', '.|').split('.'):
                    part = part.replace('.|', '. ').strip()
                    if part:
                        sentences.append(part)
                
                if not sentences:
                    sentences = [txt[:500]]  # fallback
                
                # Group sentences into chunks that fit under max_tokens
                chunks = []
                current_chunk = ""
                
                for sent in sentences:
                    test_chunk = (current_chunk + " " + sent).strip()
                    # Try to tokenize to see if it fits under the limit
                    try:
                        _ = tokenize([test_chunk])
                        current_chunk = test_chunk
                    except RuntimeError:
                        # Too long, save current chunk and start new
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If still too long, truncate each chunk to first N characters
                safe_chunks = []
                for chunk in chunks:
                    while len(chunk) > 50:  # at least some words
                        try:
                            _ = tokenize([chunk])
                            safe_chunks.append(chunk)
                            break
                        except RuntimeError:
                            # Reduce by 20%
                            chunk = chunk[:int(len(chunk) * 0.8)]
                    else:
                        if chunk:
                            safe_chunks.append(chunk)
                
                if not safe_chunks:
                    # Last fallback: first 200 characters
                    safe_chunks = [txt[:200]]
                
                # Process chunks and average
                all_chunk_feats = []
                for chunk in safe_chunks:
                    with torch.no_grad():
                        feats = model.encode_text([[chunk]])
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        all_chunk_feats.append(feats.cpu().numpy().astype(np.float32)[0])
                
                # Average of all chunks
                feats_dict[name] = np.mean(all_chunk_feats, axis=0)
                
        except Exception as ex:
            failures[name] = f"{type(ex).__name__}: {str(ex)[:100]}"
            continue
    
    return feats_dict, failures

def l2_normalize_rows(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def extract_clip_features(valid_names, img_names, txt_names, device, batch_size=32):
    """Extract features with CLIP (text + images)"""
    print("\n" + "="*80)
    print("EXTRACTION WITH CLIP (text + images)")
    print("="*80)
    
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        print("Trying without use_safetensors...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    clip_model.to(device)
    clip_model.eval()
    
    # Images
    print("Extracting image features with CLIP...")
    image_feats = []
    image_paths = [img_names[n] for n in valid_names]
    img_failures = 0
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert('RGB'))
            except Exception as e:
                if img_failures < 5:
                    print(f"\nError loading {p}: {e}")
                img_failures += 1
                images.append(Image.new('RGB', (224, 224)))
        
        try:
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                image_feats.append(feats.cpu().numpy())
        except Exception as e:
            print(f"\nError batch images {i}: {e}")
            # Fallback: zero vectors
            image_feats.append(np.zeros((len(images), 512), dtype=np.float32))
    
    if img_failures > 0:
        print(f"\n{img_failures} images generated errors")
    
    image_clip = np.vstack(image_feats).astype(np.float32)
    
    # Texts
    print("Extracting text features with CLIP...")
    text_feats = []
    text_paths = [txt_names[n] for n in valid_names]
    txt_failures = 0
    
    for i in tqdm(range(0, len(text_paths), batch_size), desc="CLIP texts"):
        batch_paths = text_paths[i:i+batch_size]
        texts = []
        for p in batch_paths:
            try:
                txt_content = Path(p).read_text(encoding='utf-8').strip()
                # CLIP has 77 token limit, so truncate more aggressively
                texts.append(txt_content[:500])
            except Exception as e:
                if txt_failures < 5:
                    print(f"\nError reading {p}: {e}")
                txt_failures += 1
                texts.append("")
        
        try:
            inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                feats = clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                text_feats.append(feats.cpu().numpy())
        except Exception as e:
            print(f"\nError batch texts {i}: {e}")
            # Fallback: zero vectors
            text_feats.append(np.zeros((len(texts), 512), dtype=np.float32))
    
    if txt_failures > 0:
        print(f"\n{txt_failures} texts generated errors")
    
    text_clip = np.vstack(text_feats).astype(np.float32)
    
    print(f"CLIP completed: images {image_clip.shape}, texts {text_clip.shape}")
    return image_clip, text_clip

def extract_minilm_features(valid_names, txt_names, batch_size=32):
    """Extract features with MiniLM (text only)"""
    print("\n" + "="*80)
    print("EXTRACTION WITH MiniLM (text only)")
    print("="*80)
    
    minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Extracting text features with MiniLM...")
    texts = []
    text_paths = [txt_names[n] for n in valid_names]
    
    for p in text_paths:
        try:
            texts.append(Path(p).read_text(encoding='utf-8').strip())
        except Exception as e:
            print(f"Error reading {p}: {e}")
            texts.append("")
    
    text_minilm = minilm.encode(texts, batch_size=batch_size, show_progress_bar=True, 
                                convert_to_numpy=True, normalize_embeddings=True)
    text_minilm = text_minilm.astype(np.float32)
    
    print(f"MiniLM completed: {text_minilm.shape}")
    return text_minilm

def extract_vggish_features(valid_names, aud_names, vggish_model, device='cpu', max_workers=1, batch_inference_size=None):
    """Extract features with VGGish PyTorch (audio only) - per-file behavior (compact)

    This version processes each file individually: for each file calls
    vggish_input.wavfile_to_examples() and then passes the examples to the model.
    Keeps the ability to specify `device` and `max_workers` (for ThreadPool)
    for compatibility with existing calls.
    """
    print("\n" + "="*80)
    print("EXTRACTION WITH VGGish PyTorch (audio only)")
    print("="*80)

    audio_feats = []
    audio_paths = [aud_names[n] for n in valid_names]
    failures = 0

    if max_workers is None:
        import multiprocessing
        max_workers = min(multiprocessing.cpu_count(), 1)

    print(f"Extracting audio features with VGGish (per-file) using {max_workers} workers...")

    from concurrent.futures import ThreadPoolExecutor

    def process_single_file(audio_path):
        try:
            examples = vggish_input.wavfile_to_examples(str(audio_path))
            if len(examples) == 0:
                return None, "No audio patches generated"

            # move examples to model device if necessary
            try:
                examples = examples.to(device)
            except Exception:
                pass

            with torch.no_grad():
                embeddings = vggish_model(examples)

            embeddings_np = embeddings.cpu().numpy()
            audio_feat = np.mean(embeddings_np, axis=0)
            norm = np.linalg.norm(audio_feat)
            if norm > 0:
                audio_feat = audio_feat / norm
            return audio_feat, None
        except Exception as e:
            return None, str(e)

    # Process in parallel batches (I/O + preprocessing)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_file, audio_paths),
            total=len(audio_paths),
            desc="VGGish audio"
        ))

    for i, (feat, error) in enumerate(results):
        if feat is not None:
            audio_feats.append(feat)
        else:
            if failures < 5:
                print(f"\nVGGish error {audio_paths[i]}: {error}")
            failures += 1
            audio_feats.append(np.zeros(128, dtype=np.float32))

    if failures > 0:
        print(f"\n{failures} audio files generated errors (replaced with zero vectors)")

    audio_vggish = np.stack(audio_feats).astype(np.float32)
    print(f"VGGish completed: {audio_vggish.shape}")
    return audio_vggish

def extract_vit_features(valid_names, img_names, device, batch_size=32):
    """Extract features with ViT (images only)"""
    print("\n" + "="*80)
    print("EXTRACTION WITH ViT (images only)")
    print("="*80)
    
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    vit_model.eval()
    
    print("Extracting image features with ViT...")
    image_feats = []
    image_paths = [img_names[n] for n in valid_names]
    img_failures = 0
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="ViT images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p)
                # Handle palette images with transparency
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA').convert('RGB')
                else:
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                if img_failures < 5:
                    print(f"\nError loading {p}: {e}")
                img_failures += 1
                images.append(Image.new('RGB', (224, 224)))
        
        try:
            inputs = vit_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = vit_model(**inputs)
                feats = outputs.last_hidden_state[:, 0, :]  # CLS token
                feats = feats / feats.norm(dim=-1, keepdim=True)
                image_feats.append(feats.cpu().numpy())
        except Exception as e:
            print(f"\nError batch ViT {i}: {e}")
            # Fallback: zero vectors (768 for ViT base)
            image_feats.append(np.zeros((len(images), 768), dtype=np.float32))
    
    if img_failures > 0:
        print(f"\n{img_failures} images generated errors in ViT")
    
    image_vit = np.vstack(image_feats).astype(np.float32)
    
    print(f"ViT completato: {image_vit.shape}")
    return image_vit


def load_or_extract_single(outdir: Path, filename: str, extract_func, *args, desc: str = "", **kwargs):
    fpath = outdir / filename
    if fpath.exists():
        print(f"\n{desc} features already exist, loading")
        arr = np.load(fpath)
        print(f"  Loaded: {arr.shape}")
        return arr
    else:
        try:
            result = extract_func(*args, **kwargs)
            np.save(fpath, result)
            print(f"Saved {desc} embeddings")
            return result
        except Exception as e:
            print(f" {desc} error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=False, help="Cartella con immagini (.jpg/.png)", default="../lastfm/raw/_images")
    parser.add_argument("--audios", required=False, help="Cartella con audio (.wav)", default="../lastfm/raw/_audios")
    parser.add_argument("--texts", required=False, help="Cartella con testi (.txt)", default="../lastfm/raw/_texts")
    parser.add_argument("--weights", default="AudioCLIP-Full-Training.pt", help="Path al file di pesi AudioCLIP (.pt). Se omesso usa pretrained=True del repo.")
    parser.add_argument("--outdir", default="lastfm_features", help="Cartella output")
    parser.add_argument("--device", default="cuda:1", help="cuda o cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=512, help="batch size per immagini/testo")
    parser.add_argument("--window-batch-size", type=int, default=512, help="batch size quando si codificano molte finestre audio")
    parser.add_argument("--audio-sr", type=int, default=44100, help="Sample rate per caricare gli audio")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Lunghezza finestra (s) per sliding mean")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="Stride (s) per sliding mean")
    parser.add_argument("--no-concat", action="store_true", help="Non creare concatenated.npy, salva solo per-modality .npy",default=True)
    parser.add_argument("--l2norm", action="store_true", help="Applica L2-normalizzazione alle righe PRIMA della concatenazione")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    try:
        ensure_bpe(repo_root)
    except Exception as ex:
        print("[error] BPE check/download failed:", ex)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"Using device: {device}")


    model = load_model(args.weights, device)
    preprocess, res = get_image_preprocess(model)
    print(f"Image preprocess resolution: {res}")

    vggish_device = device if device.type == 'cuda' else torch.device('cpu')
    vggish_model = load_vggish_model(vggish_device)


    img_files = sorted([str(p) for p in Path(args.images).glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    audio_files = sorted([str(p) for p in Path(args.audios).glob("*.wav")])  # Usa solo WAV
    text_files = sorted([str(p) for p in Path(args.texts).glob("*.txt")])

   
    if audio_files:
        audio_stems_valid, audio_stems_invalid = validate_all_audio_files(
            audio_files, 
            validation_mode='quick' 
        )
        audio_files = [p for p in audio_files if Path(p).stem in audio_stems_valid]
        
        if audio_stems_invalid:
            print(f"\nRemoved {len(audio_stems_invalid)} corrupted audio files from processing")
    else:
        audio_stems_invalid = {}

    img_names = {Path(p).stem: p for p in img_files}
    aud_names = {Path(p).stem: p for p in audio_files}
    txt_names = {Path(p).stem: p for p in text_files}
    common = sorted(list(set(img_names.keys()) & set(aud_names.keys()) & set(txt_names.keys())))
    if len(common) == 0:
        print("\n Warning: no files with common basename found across the three folders.")
        sys.exit(1)
    image_items = [(n, img_names[n]) for n in common]
    audio_items = [(n, aud_names[n]) for n in common]
    text_items = [(n, txt_names[n]) for n in common]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check if AudioCLIP features already exist
    audioclip_files_exist = (
        (outdir / "image_audioclip.npy").exists() and
        (outdir / "audio_audioclip.npy").exists() and
        (outdir / "text_audioclip.npy").exists()
    )
    
    csv_exists = (outdir / "item_features.csv").exists()
    
    # =============================================================================
    # Reconciliation if corrupted audio files were removed
    # =============================================================================
    if audio_stems_invalid and audioclip_files_exist:
        print("CORRUPTED AUDIO REMOVED - EMBEDDINGS RECONCILIATION")
        print("="*80)
        print(f"Corrupted audio detected by validation: {len(audio_stems_invalid)}")
        print(f"Examples: {sorted(list(audio_stems_invalid.keys()))[:5]}")
        print("\nAutomatic reconciliation of .npy and CSV files in progress...")

    if audioclip_files_exist:
        print("\n" + "="*80)
        print("AudioCLIP features already exist, loading...")
        print("="*80)
        images_np = np.load(outdir / "image_audioclip.npy")
        audios_np = np.load(outdir / "audio_audioclip.npy")
        texts_np = np.load(outdir / "text_audioclip.npy")
        
        # Load valid names from CSV (if exists)
        import csv
        if csv_exists:
            valid_names = []
            with open(outdir / "item_features.csv", "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    valid_names.append(row['item_id'])
            print(f"Loaded {len(valid_names)} existing samples from CSV")
        else:
            # CSV missing: reconstruct from file intersection
            print("CSV missing, reconstructing order from files...")
            # Use 'common' already calculated above - same order as first extraction
            # Verify that the number matches
            expected_count = images_np.shape[0]
            if len(common) != expected_count:
                print(f" ERROR: Number of common files ({len(common)}) != existing embeddings ({expected_count})")
                print("Delete the .npy files and rerun the full extraction.")
                sys.exit(1)
            valid_names = common
            print(f"Reconstructed {len(valid_names)} names from file order")
            
            # Save reconstructed CSV
            with open(outdir / "item_features.csv", "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["item_id", "idx"])
                for i, name in enumerate(valid_names):
                    writer.writerow([name, i])
            print(f"Saved reconstructed CSV: {outdir/'item_features.csv'}")
        
        print(f"  - image_audioclip.npy: {images_np.shape}")
        print(f"  - audio_audioclip.npy: {audios_np.shape}")
        print(f"  - text_audioclip.npy: {texts_np.shape}")
        
        # =============================================================================
        # AUTOMATIC RECONCILIATION: remove embeddings for missing files (both corrupted and absent)
        # =============================================================================
        common_set = set(common)
        valid_names_set = set(valid_names)
        
        # Find files that were present but are now missing
        # Include both corrected audio and files removed for any other reason
        missing_from_current = valid_names_set - common_set
        
        if missing_from_current:
            print("\n" + "="*80)
            print("AUTOMATIC RECONCILIATION: Missing files detected")
            print("="*80)
            print(f"Files present in saved embeddings: {len(valid_names)}")
            print(f"Common files found now: {len(common)}")
            print(f"Files to remove: {len(missing_from_current)}")
            print(f"Examples of missing files: {sorted(list(missing_from_current))[:10]}")
            
            # Identify how many are corrupted audio
            audio_corrotti_rimossi = len([n for n in missing_from_current if n in audio_stems_invalid])
            if audio_corrotti_rimossi > 0:
                print(f"\n  Among these, {audio_corrotti_rimossi} corrupted audio removed by validation")
            
            # Create mapping old_idx -> item_id and identify indices to keep
            indices_to_keep = []
            new_valid_names = []
            
            for idx, item_id in enumerate(valid_names):
                if item_id in common_set:
                    indices_to_keep.append(idx)
                    new_valid_names.append(item_id)
            
            print(f"\nKeeping {len(indices_to_keep)} rows, removing {len(missing_from_current)} rows")
            
            # Filter numpy arrays
            images_np = images_np[indices_to_keep]
            audios_np = audios_np[indices_to_keep]
            texts_np = texts_np[indices_to_keep]
            
            # Aggiorna valid_names
            valid_names = new_valid_names
            
            # Save updated embeddings
            print("\nSaving updated AudioCLIP embeddings...")
            np.save(outdir / "image_audioclip.npy", images_np)
            np.save(outdir / "audio_audioclip.npy", audios_np)
            np.save(outdir / "text_audioclip.npy", texts_np)
            print(f"  - image_audioclip.npy: {images_np.shape}")
            print(f"  - audio_audioclip.npy: {audios_np.shape}")
            print(f"  - text_audioclip.npy: {texts_np.shape}")
            
            # Save updated CSV
            with open(outdir / "item_features.csv", "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["item_id", "idx"])
                for i, name in enumerate(valid_names):
                    writer.writerow([name, i])
            print(f"Saved updated CSV: {outdir/'item_features.csv'} ({len(valid_names)} rows)")
            
            # Reconcile additional models if they exist
            print("\nReconciling additional model embeddings...")
            models_to_reconcile = [
                ("image_clip.npy", "CLIP images"),
                ("text_clip.npy", "CLIP text"),
                ("text_minilm.npy", "MiniLM"),
                ("audio_vggish.npy", "VGGish"),
                ("image_vit.npy", "ViT")
            ]
            
            for fname, desc in models_to_reconcile:
                fpath = outdir / fname
                if fpath.exists():
                    arr = np.load(fpath)
                    if arr.shape[0] == len(valid_names) + len(missing_from_current):
                        # Filter this too
                        arr_filtered = arr[indices_to_keep]
                        np.save(fpath, arr_filtered)
                        print(f"  - {fname}: {arr.shape} → {arr_filtered.shape} ({desc})")
                    elif arr.shape[0] == len(valid_names):
                        print(f"  - {fname}: already aligned ({desc})")
                    else:
                        print(f"  {fname}: incompatible dimensions, skip ({desc})")
            
            print("="*80)
        else:
            print("\nAll files are present, no reconciliation necessary")
    else:
        print("Extracting image features...")
        image_results, image_failures = process_images_ordered(model, preprocess, image_items, device, batch_size=args.batch_size)
        summarize_failures("images", image_failures)

        print("Extracting audio features (sliding_mean)...")
        audio_results, audio_failures = process_audios_sliding_mean(model, audio_items, device,
                                               sr=args.audio_sr,
                                               window_sec=args.window_sec,
                                               stride_sec=args.stride_sec,
                                               window_batch_size=args.window_batch_size)
        summarize_failures("audio", audio_failures)

        print("Extracting text features...")
        text_results, text_failures = process_texts_ordered(model, text_items, device, batch_size=max(8, args.batch_size))
        summarize_failures("texts", text_failures)

        valid_names = [n for n in common if n in image_results and n in audio_results and n in text_results]
        dropped = [n for n in common if n not in valid_names]
        
        print("\n" + "="*80)
        print("FINAL EXTRACTION REPORT")
        print("="*80)
        print(f"Initial samples with all 3 modalities: {len(common)}")
        print(f"Samples extracted successfully: {len(valid_names)}")
        
        if dropped:
            print(f"\nRemoved {len(dropped)} samples due to extraction errors:")
            print(f"   Examples: {dropped[:10]}")
            
            # Detail errors by modality
            dropped_img = [n for n in dropped if n not in image_results]
            dropped_aud = [n for n in dropped if n not in audio_results]
            dropped_txt = [n for n in dropped if n not in text_results]
            
            if dropped_img:
                print(f"   - {len(dropped_img)} failed for image")
            if dropped_aud:
                print(f"   - {len(dropped_aud)} failed for audio")
            if dropped_txt:
                print(f"   - {len(dropped_txt)} failed for text")
        
        if not valid_names:
            print("\n ERROR: No embeddings extracted successfully. Check previous messages.")
            sys.exit(1)
        
        print(f"\nProceeding with {len(valid_names)} valid samples")
        print("="*80 + "\n")

        images_np = np.stack([image_results[n] for n in valid_names], axis=0).astype(np.float32)
        audios_np = np.stack([audio_results[n] for n in valid_names], axis=0).astype(np.float32)
        texts_np = np.stack([text_results[n] for n in valid_names], axis=0).astype(np.float32)

        # Save AudioCLIP embeddings
        np.save(outdir / "image_audioclip.npy", images_np)
        print(f"Saved {outdir/'image_audioclip.npy'} shape={images_np.shape}")
        np.save(outdir / "audio_audioclip.npy", audios_np)
        print(f"Saved {outdir/'audio_audioclip.npy'} shape={audios_np.shape}")
        np.save(outdir / "text_audioclip.npy", texts_np)
        print(f"Saved {outdir/'text_audioclip.npy'} shape={texts_np.shape}")
        
        # Save CSV mapping
        import csv
        with open(outdir / "item_features.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "idx"])
            for i, name in enumerate(valid_names):
                writer.writerow([name, i])
        print(f"Saved mapping {outdir/'item_features.csv'} N={len(valid_names)}")

    # =============================================================================
    # EXTRACTION WITH OTHER MODELS (same valid_names samples)
    # =============================================================================
    
    print("\n" + "="*80)
    print("START EXTRACTION WITH ADDITIONAL MODELS")
    print("="*80)
    
    if (outdir / "image_clip.npy").exists() and (outdir / "text_clip.npy").exists():
        print("\nCLIP features already exist, skip extraction")
        image_clip = np.load(outdir / "image_clip.npy")
        text_clip = np.load(outdir / "text_clip.npy")
        print(f"  Loaded: images {image_clip.shape}, texts {text_clip.shape}")
    else:
        try:
            image_clip, text_clip = extract_clip_features(valid_names, img_names, txt_names, device, batch_size=args.batch_size)
            np.save(outdir / "image_clip.npy", image_clip)
            np.save(outdir / "text_clip.npy", text_clip)
            print(f"Saved CLIP embeddings")
        except Exception as e:
            print(f" CLIP error: {e}")
    
    # MiniLM (text only)
    text_minilm = load_or_extract_single(outdir, "text_minilm.npy", extract_minilm_features, valid_names, txt_names, batch_size=args.batch_size, desc="MiniLM")
    
    # ViT (images only) - run before VGGish for image priority
    image_vit = load_or_extract_single(outdir, "image_vit.npy", extract_vit_features, valid_names, img_names, device, batch_size=args.batch_size, desc="ViT")


    # VGGish (audio only) - replaces Whisper; run after ViT
    audio_vggish = load_or_extract_single(outdir, "audio_vggish.npy", extract_vggish_features, valid_names, aud_names, vggish_model, device=vggish_device, max_workers=1, batch_inference_size=1, desc="VGGish")


    # optional L2 normalization before concatenation
    if args.l2norm:
        images_np = l2_normalize_rows(images_np)
        audios_np = l2_normalize_rows(audios_np)
        texts_np = l2_normalize_rows(texts_np)

    # concatenation if requested
    if not args.no_concat:
        concatenated = np.concatenate([images_np.astype(np.float32), audios_np.astype(np.float32), texts_np.astype(np.float32)], axis=1)
        np.save(outdir / "concatenated.npy", concatenated)
        print(f"Saved {outdir/'concatenated.npy'} shape={concatenated.shape}")

    # Save CSV mapping if not exists
    if not (outdir / "item_features.csv").exists():
        import csv
        with open(outdir / "item_features.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "idx"])
            for i, name in enumerate(valid_names):
                writer.writerow([name, i])
        print(f"Saved mapping {outdir/'item_features.csv'} N={len(valid_names)}")

    print("\n" + "="*80)
    print("EXTRACTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Files saved in: {outdir}")
    print(f"\nAudioCLIP:")
    print(f"  - image_audioclip.npy:  {images_np.shape}")
    print(f"  - audio_audioclip.npy:  {audios_np.shape}")
    print(f"  - text_audioclip.npy:   {texts_np.shape}")
    print(f"\nAdditional models (if available):")
    
    # Check and print info on saved files
    all_files = {
        "image_clip.npy": "CLIP images",
        "text_clip.npy": "CLIP text",
        "text_minilm.npy": "MiniLM text",
        "audio_vggish.npy": "VGGish audio",
        "image_vit.npy": "ViT images"
    }
    
    for fname, desc in all_files.items():
        fpath = outdir / fname
        if fpath.exists():
            arr = np.load(fpath)
            print(f"  - {fname}: {arr.shape} ({desc})")
    
    if not args.no_concat:
        print(f"\nConcatenated:")
        print(f"  - concatenated.npy: {concatenated.shape}")
    print(f"\nMapping:")
    print(f"  - item_features.csv: {len(valid_names)} rows")
    print("="*80)

if __name__ == "__main__":
    main()