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
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel

# VGGish PyTorch
from torchvggish import vggish, vggish_input

# URL BPE (raw gz) dal repo CLIP ufficiale
BPE_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
BPE_LOCAL_REL = "utils/bpe_simple_vocab_16e6.txt.gz"

def ensure_bpe(repo_root: Path):
    """
    Assicura che utils/bpe_simple_vocab_16e6.txt.gz esista e sia un gzip valido.
    In caso contrario, prova a scaricarlo dal repo CLIP.
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
    Valida un file audio con librosa (veloce, basico).
    Restituisce (is_valid, error_message)
    """
    try:
        # Prova a caricare l'audio
        y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
        
        # Verifica che l'audio non sia vuoto
        if len(y) == 0:
            return False, "Audio vuoto (0 campioni)"
        
        # Verifica sample rate
        if sr_loaded <= 0:
            return False, f"Invalid sample rate: {sr_loaded}"
        
        return True, None
    except librosa.LibrosaError as e:
        return False, f"LibrosaError: {str(e)[:50]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:50]}"

def validate_vggish_audio_file(audio_path):
    """
    Valida un file audio specificamente per VGGish (wavfile_to_examples).
    Restituisce (is_valid, error_message)
    """
    try:
        # Prova a processare con vggish_input
        # Nota: vggish_input.wavfile_to_examples richiede che il file sia in formato
        # WAV valido con sample rate corretto (tipicamente 16kHz)
        examples = vggish_input.wavfile_to_examples(str(audio_path))
        
        # Verifica che gli esempi non siano vuoti
        if len(examples) == 0:
            return False, "VGGish generated no patches"
        
        return True, None
    except TypeError as e:
        # Errore tipico: 'float' object cannot be interpreted as an integer
        # Significa che il file WAV ha un formato non standard
        return False, f"Invalid WAV format: {str(e)[:40]}"
    except Exception as e:
        return False, f"VGGish error: {str(e)[:50]}"

def validate_all_audio_files(audio_files, validation_mode='quick'):
    """
    Valida TUTTI i file audio e restituisce:
    - valid_stems: lista di stem validi
    - invalid_stems: dict {stem: error_reason}
    
    validation_mode:
      'quick': usa solo librosa.load (veloce)
      'vggish': usa vggish_input.wavfile_to_examples (più severo, ma lento)
    """
    print("\n" + "="*80)
    print("VALIDAZIONE PRELIMINARE FILE AUDIO")
    print("="*80)
    print(f"Controllo {len(audio_files)} file audio ({validation_mode} mode)...")
    
    valid_stems = []
    invalid_stems = {}
    
    for audio_path in tqdm(audio_files, desc="Audio validation"):
        stem = Path(audio_path).stem
        
        if validation_mode == 'vggish':
            is_valid, error = validate_vggish_audio_file(audio_path)
        else:
            is_valid, error = validate_audio_file(audio_path)
        
        if is_valid:
            valid_stems.append(stem)
        else:
            invalid_stems[stem] = error
    
    print(f"\n✓ Validazione completata:")
    print(f"  - File validi: {len(valid_stems)}")
    print(f"  - File corrotti: {len(invalid_stems)}")
    
    if invalid_stems:
        print(f"\n⚠ File audio corrotti trovati:")
        for stem, error in list(invalid_stems.items())[:10]:
            print(f"  - {stem}: {error}")
        if len(invalid_stems) > 10:
            print(f"  ... e altri {len(invalid_stems) - 10} file corrotti")
        
        # Salva log dei file corrotti
        log_path = Path("corrupted_audios.log")
        with open(log_path, "w", encoding='utf-8') as f:
            f.write(f"Validazione audio: {len(invalid_stems)} file corrotti\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for stem, error in sorted(invalid_stems.items()):
                f.write(f"{stem}: {error}\n")
        print(f"\n✓ Log salvato: {log_path}")
    else:
        print(f"\n✓ Tutti i file audio sono validi!")
    
    print("="*80)
    
    return valid_stems, invalid_stems

def summarize_failures(label: str, failures: Dict[str, str], max_items: int = 5) -> None:
    """Logga un riepilogo leggibile dei file saltati."""
    if not failures:
        return
    print(f"[warn] {len(failures)} {label} saltati per errori di estrazione.")
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

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def load_model(weights, device):
    # import here after the BPE has been ensured
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
    Carica il modello VGGish usando torchvggish (PyTorch).
    Restituisce il modello PyTorch pronto per l'inferenza.
    """
    print("[VGGish] Caricamento modello torchvggish (PyTorch)...")
    model = vggish()
    model.eval()
    # Usa device specificato (GPU se disponibile)
    model.to(device)
    print(f"[VGGish] ✓ Modello caricato con successo su {device}")
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
                raise ValueError("audio vuoto")
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
    Processa i testi con sliding window se necessario.
    Se un testo supera max_tokens dopo tokenizzazione, lo divide in chunk e fa la media degli embeddings.
    """
    feats_dict: Dict[str, np.ndarray] = {}
    failures: Dict[str, str] = {}
    
    # Import tokenizer from the model
    from model.clip.clip import tokenize
    
    for name, path in text_items:
        try:
            txt = Path(path).read_text(encoding='utf-8').strip()
            
            # Tokenizza il testo per vedere quanto è lungo
            try:
                # Prova prima a tokenizzare normalmente
                with torch.no_grad():
                    feats = model.encode_text([[txt]])
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    feats_dict[name] = feats.cpu().numpy().astype(np.float32)[0]
            except RuntimeError as e:
                if "too long for context length" not in str(e):
                    raise
                
                # Il testo è troppo lungo, dividiamolo in frasi
                # Splitta per punti, mantieni i separatori
                sentences = []
                for part in txt.replace('! ', '.|').replace('? ', '.|').split('.'):
                    part = part.replace('.|', '. ').strip()
                    if part:
                        sentences.append(part)
                
                if not sentences:
                    sentences = [txt[:500]]  # fallback
                
                # Raggruppa le frasi in chunk che stanno sotto max_tokens
                chunks = []
                current_chunk = ""
                
                for sent in sentences:
                    test_chunk = (current_chunk + " " + sent).strip()
                    # Prova a tokenizzare per vedere se sta sotto il limite
                    try:
                        _ = tokenize([test_chunk])
                        current_chunk = test_chunk
                    except RuntimeError:
                        # Troppo lungo, salva il chunk corrente e inizia nuovo
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Se ancora troppo lunghi, tronca ogni chunk ai primi N caratteri
                safe_chunks = []
                for chunk in chunks:
                    while len(chunk) > 50:  # almeno qualche parola
                        try:
                            _ = tokenize([chunk])
                            safe_chunks.append(chunk)
                            break
                        except RuntimeError:
                            # Riduci del 20%
                            chunk = chunk[:int(len(chunk) * 0.8)]
                    else:
                        if chunk:
                            safe_chunks.append(chunk)
                
                if not safe_chunks:
                    # Ultimo fallback: primi 200 caratteri
                    safe_chunks = [txt[:200]]
                
                # Processa i chunk e fai la media
                all_chunk_feats = []
                for chunk in safe_chunks:
                    with torch.no_grad():
                        feats = model.encode_text([[chunk]])
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        all_chunk_feats.append(feats.cpu().numpy().astype(np.float32)[0])
                
                # Media di tutti i chunk
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
    """Estrae feature con CLIP (testo + immagini)"""
    print("\n" + "="*80)
    print("ESTRAZIONE CON CLIP (testo + immagini)")
    print("="*80)
    
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"Errore caricamento modello CLIP: {e}")
        print("Provo senza use_safetensors...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    clip_model.to(device)
    clip_model.eval()
    
    # Immagini
    print("Estraggo feature immagini con CLIP...")
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
                    print(f"\nErrore caricamento {p}: {e}")
                img_failures += 1
                images.append(Image.new('RGB', (224, 224)))
        
        try:
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                image_feats.append(feats.cpu().numpy())
        except Exception as e:
            print(f"\nErrore batch immagini {i}: {e}")
            # Fallback: zero vectors
            image_feats.append(np.zeros((len(images), 512), dtype=np.float32))
    
    if img_failures > 0:
        print(f"\n⚠ {img_failures} immagini hanno generato errori")
    
    image_clip = np.vstack(image_feats).astype(np.float32)
    
    # Testi
    print("Estraggo feature testo con CLIP...")
    text_feats = []
    text_paths = [txt_names[n] for n in valid_names]
    txt_failures = 0
    
    for i in tqdm(range(0, len(text_paths), batch_size), desc="CLIP texts"):
        batch_paths = text_paths[i:i+batch_size]
        texts = []
        for p in batch_paths:
            try:
                txt_content = Path(p).read_text(encoding='utf-8').strip()
                # CLIP ha limite di 77 token, quindi tronca più aggressivamente
                texts.append(txt_content[:500])
            except Exception as e:
                if txt_failures < 5:
                    print(f"\nErrore lettura {p}: {e}")
                txt_failures += 1
                texts.append("")
        
        try:
            inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                feats = clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                text_feats.append(feats.cpu().numpy())
        except Exception as e:
            print(f"\nErrore batch testi {i}: {e}")
            # Fallback: zero vectors
            text_feats.append(np.zeros((len(texts), 512), dtype=np.float32))
    
    if txt_failures > 0:
        print(f"\n⚠ {txt_failures} testi hanno generato errori")
    
    text_clip = np.vstack(text_feats).astype(np.float32)
    
    print(f"✓ CLIP completato: images {image_clip.shape}, texts {text_clip.shape}")
    return image_clip, text_clip

def extract_minilm_features(valid_names, txt_names, batch_size=32):
    """Estrae feature con MiniLM (solo testo)"""
    print("\n" + "="*80)
    print("ESTRAZIONE CON MiniLM (solo testo)")
    print("="*80)
    
    minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Estraggo feature testo con MiniLM...")
    texts = []
    text_paths = [txt_names[n] for n in valid_names]
    
    for p in text_paths:
        try:
            texts.append(Path(p).read_text(encoding='utf-8').strip())
        except Exception as e:
            print(f"Errore lettura {p}: {e}")
            texts.append("")
    
    text_minilm = minilm.encode(texts, batch_size=batch_size, show_progress_bar=True, 
                                convert_to_numpy=True, normalize_embeddings=True)
    text_minilm = text_minilm.astype(np.float32)
    
    print(f"✓ MiniLM completato: {text_minilm.shape}")
    return text_minilm

def extract_vggish_features(valid_names, aud_names, vggish_model, device='cpu', max_workers=8, batch_inference_size=None):
    """Estrae feature con VGGish PyTorch (solo audio) - comportamento per-file (compatto)

    Questa versione elabora ogni file singolarmente: per ogni file chiama
    vggish_input.wavfile_to_examples() e poi passa gli esempi al modello.
    Mantiene la possibilità di specificare `device` e `max_workers` (per ThreadPool)
    per compatibilità con le chiamate esistenti.
    """
    print("\n" + "="*80)
    print("ESTRAZIONE CON VGGish PyTorch (solo audio)")
    print("="*80)

    audio_feats = []
    audio_paths = [aud_names[n] for n in valid_names]
    failures = 0

    # Imposta default per max_workers se None
    if max_workers is None:
        import multiprocessing
        max_workers = min(multiprocessing.cpu_count(), 8)

    print(f"Estraggo feature audio con VGGish (per-file) usando {max_workers} worker...")

    from concurrent.futures import ThreadPoolExecutor

    def process_single_file(audio_path):
        try:
            examples = vggish_input.wavfile_to_examples(str(audio_path))
            if len(examples) == 0:
                return None, "No audio patches generated"

            # sposta gli esempi sul device del modello se necessario
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

    # Processa in batch paralleli (I/O + preprocessing)
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
                print(f"\nErrore VGGish {audio_paths[i]}: {error}")
            failures += 1
            audio_feats.append(np.zeros(128, dtype=np.float32))

    if failures > 0:
        print(f"\n⚠ {failures} file audio hanno generato errori (sostituiti con zero vectors)")

    audio_vggish = np.stack(audio_feats).astype(np.float32)
    print(f"✓ VGGish completato: {audio_vggish.shape}")
    return audio_vggish

def extract_vit_features(valid_names, img_names, device, batch_size=32):
    """Estrae feature con ViT (solo immagini)"""
    print("\n" + "="*80)
    print("ESTRAZIONE CON ViT (solo immagini)")
    print("="*80)
    
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    vit_model.eval()
    
    print("Estraggo feature immagini con ViT...")
    image_feats = []
    image_paths = [img_names[n] for n in valid_names]
    img_failures = 0
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="ViT images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p)
                # Gestisci palette images con trasparenza
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA').convert('RGB')
                else:
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                if img_failures < 5:
                    print(f"\nErrore caricamento {p}: {e}")
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
            print(f"\nErrore batch ViT {i}: {e}")
            # Fallback: zero vectors (768 per ViT base)
            image_feats.append(np.zeros((len(images), 768), dtype=np.float32))
    
    if img_failures > 0:
        print(f"\n⚠ {img_failures} immagini hanno generato errori in ViT")
    
    image_vit = np.vstack(image_feats).astype(np.float32)
    
    print(f"✓ ViT completato: {image_vit.shape}")
    return image_vit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=False, help="Cartella con immagini (.jpg/.png)", default="../lastfm/raw/_images")
    parser.add_argument("--audios", required=False, help="Cartella con audio (.wav)", default="../lastfm/raw/_audios")
    parser.add_argument("--texts", required=False, help="Cartella con testi (.txt)", default="../lastfm/raw/_texts")
    parser.add_argument("--weights", default="AudioCLIP-Full-Training.pt", help="Path al file di pesi AudioCLIP (.pt). Se omesso usa pretrained=True del repo.")
    parser.add_argument("--outdir", default="lastfm_features", help="Cartella output")
    parser.add_argument("--device", default="cuda", help="cuda o cpu (default: auto)")
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
        print("Se preferisci, applica la patch al tokenizer o scarica manualmente il file BPE gz nella cartella utils/")
        sys.exit(1)

    # Gestione device con fallback automatico
    if args.device == "cuda":
        try:
            # Tenta di inizializzare CUDA
            if torch.cuda.is_available():
                torch.cuda.init()
                device = torch.device("cuda")
                print(f"Device: cuda (GPU: {torch.cuda.get_device_name(0)})")
            else:
                print("[warn] CUDA richiesto ma non disponibile, fallback a CPU")
                device = torch.device("cpu")
        except Exception as e:
            print(f"[warn] Errore inizializzazione CUDA: {e}")
            print("[warn] Fallback automatico a CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        print(f"Device: {device}")

    # carica modello AudioCLIP
    model = load_model(args.weights, device)
    preprocess, res = get_image_preprocess(model)
    print(f"Image preprocess resolution: {res}")
    
    # Carica VGGish - usa GPU se disponibile per velocità
    vggish_device = device if device.type == 'cuda' else torch.device('cpu')
    vggish_model = load_vggish_model(vggish_device)

    # lista file e matching per basename
    img_files = sorted([str(p) for p in Path(args.images).glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    audio_files = sorted([str(p) for p in Path(args.audios).glob("*.wav")])  # Usa solo WAV
    text_files = sorted([str(p) for p in Path(args.texts).glob("*.txt")])

    print(f"File trovati (grezzo): {len(img_files)} immagini, {len(audio_files)} audio, {len(text_files)} testi")

    # =========================================================================
    # VALIDAZIONE PRELIMINARE AUDIO - Rimuovi file corrotti PRIMA di processare
    # =========================================================================
    if audio_files:
        audio_stems_valid, audio_stems_invalid = validate_all_audio_files(
            audio_files, 
            validation_mode='quick'  # usa librosa.load (veloce)
            # Cambia a 'vggish' per test più severi, ma è più lento
        )
        # Filtra audio_files per mantenere solo i validi
        audio_files = [p for p in audio_files if Path(p).stem in audio_stems_valid]
        
        if audio_stems_invalid:
            print(f"\n⚠ Rimossi {len(audio_stems_invalid)} file audio corrotti dall'elaborazione")
    else:
        audio_stems_invalid = {}

    img_names = {Path(p).stem: p for p in img_files}
    aud_names = {Path(p).stem: p for p in audio_files}
    txt_names = {Path(p).stem: p for p in text_files}

    print(f"\nFile dopo validazione: {len(img_files)} immagini, {len(audio_files)} audio, {len(text_files)} testi")

    common = sorted(list(set(img_names.keys()) & set(aud_names.keys()) & set(txt_names.keys())))
    if len(common) == 0:
        print("\n❌ Attenzione: nessun file con basename comune trovato tra le tre cartelle.")
        sys.exit(1)
    print(f"\n✓ Trovati {len(common)} campioni con TUTTE E 3 le modalità. Esempio: {common[:5]}")

    image_items = [(n, img_names[n]) for n in common]
    audio_items = [(n, aud_names[n]) for n in common]
    text_items = [(n, txt_names[n]) for n in common]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Controlla se le feature AudioCLIP esistono già
    audioclip_files_exist = (
        (outdir / "image_audioclip.npy").exists() and
        (outdir / "audio_audioclip.npy").exists() and
        (outdir / "text_audioclip.npy").exists()
    )
    
    csv_exists = (outdir / "item_features.csv").exists()
    
    # =============================================================================
    # Riconciliazione se audio corrotti sono stati rimossi
    # =============================================================================
    if audio_stems_invalid and audioclip_files_exist:
        print("\n" + "="*80)
        print("⚠ AUDIO CORROTTI RIMOSSI - RICONCILIAZIONE EMBEDDINGS")
        print("="*80)
        print(f"Audio corrotti rilevati dalla validazione: {len(audio_stems_invalid)}")
        print(f"Esempi: {sorted(list(audio_stems_invalid.keys()))[:5]}")
        print("\nRiconciliazione automatica dei file .npy e CSV in corso...")

    if audioclip_files_exist:
        print("\n" + "="*80)
        print("✓ Feature AudioCLIP già esistenti, caricamento in corso...")
        print("="*80)
        images_np = np.load(outdir / "image_audioclip.npy")
        audios_np = np.load(outdir / "audio_audioclip.npy")
        texts_np = np.load(outdir / "text_audioclip.npy")
        
        # Carica i nomi validi dal CSV (se esiste)
        import csv
        if csv_exists:
            valid_names = []
            with open(outdir / "item_features.csv", "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    valid_names.append(row['item_id'])
            print(f"✓ Caricati {len(valid_names)} campioni esistenti dal CSV")
        else:
            # CSV mancante: ricostruisci dall'intersezione dei file
            print("⚠ CSV mancante, ricostruzione dell'ordine dai file...")
            # Usa 'common' già calcolato sopra - stesso ordine della prima estrazione
            # Verifica che il numero corrisponda
            expected_count = images_np.shape[0]
            if len(common) != expected_count:
                print(f"❌ ERRORE: Numero file comuni ({len(common)}) != embeddings esistenti ({expected_count})")
                print("   Elimina i file .npy e riesegui l'estrazione completa.")
                sys.exit(1)
            valid_names = common
            print(f"✓ Ricostruiti {len(valid_names)} nomi dall'ordine dei file")
            
            # Salva il CSV ricostruito
            with open(outdir / "item_features.csv", "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["item_id", "idx"])
                for i, name in enumerate(valid_names):
                    writer.writerow([name, i])
            print(f"✓ Salvato CSV ricostruito: {outdir/'item_features.csv'}")
        
        print(f"  - image_audioclip.npy: {images_np.shape}")
        print(f"  - audio_audioclip.npy: {audios_np.shape}")
        print(f"  - text_audioclip.npy: {texts_np.shape}")
        
        # =============================================================================
        # RICONCILIAZIONE AUTOMATICA: rimuovi embeddings per file mancanti (sia corrotti che assenti)
        # =============================================================================
        common_set = set(common)
        valid_names_set = set(valid_names)
        
        # Trova file che erano presenti ma ora mancano
        # Includi sia audio corretti che file rimossi per qualsiasi altro motivo
        missing_from_current = valid_names_set - common_set
        
        if missing_from_current:
            print("\n" + "="*80)
            print("⚠ RICONCILIAZIONE AUTOMATICA: File mancanti rilevati")
            print("="*80)
            print(f"File presenti negli embeddings salvati: {len(valid_names)}")
            print(f"File comuni trovati ora: {len(common)}")
            print(f"File da rimuovere: {len(missing_from_current)}")
            print(f"Esempi di file mancanti: {sorted(list(missing_from_current))[:10]}")
            
            # Identifica quanti sono audio corrotti
            audio_corrotti_rimossi = len([n for n in missing_from_current if n in audio_stems_invalid])
            if audio_corrotti_rimossi > 0:
                print(f"\n  Tra questi, {audio_corrotti_rimossi} audio corrotti e rimossi dalla validazione")
            
            # Crea mapping old_idx -> item_id e identifica indici da mantenere
            indices_to_keep = []
            new_valid_names = []
            
            for idx, item_id in enumerate(valid_names):
                if item_id in common_set:
                    indices_to_keep.append(idx)
                    new_valid_names.append(item_id)
            
            print(f"\n✓ Mantengo {len(indices_to_keep)} righe, rimuovo {len(missing_from_current)} righe")
            
            # Filtra gli array numpy
            images_np = images_np[indices_to_keep]
            audios_np = audios_np[indices_to_keep]
            texts_np = texts_np[indices_to_keep]
            
            # Aggiorna valid_names
            valid_names = new_valid_names
            
            # Salva gli embeddings aggiornati
            print("\n✓ Salvataggio embeddings AudioCLIP aggiornati...")
            np.save(outdir / "image_audioclip.npy", images_np)
            np.save(outdir / "audio_audioclip.npy", audios_np)
            np.save(outdir / "text_audioclip.npy", texts_np)
            print(f"  - image_audioclip.npy: {images_np.shape}")
            print(f"  - audio_audioclip.npy: {audios_np.shape}")
            print(f"  - text_audioclip.npy: {texts_np.shape}")
            
            # Salva CSV aggiornato
            with open(outdir / "item_features.csv", "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["item_id", "idx"])
                for i, name in enumerate(valid_names):
                    writer.writerow([name, i])
            print(f"✓ Salvato CSV aggiornato: {outdir/'item_features.csv'} ({len(valid_names)} righe)")
            
            # Riconcilia anche gli altri modelli se esistono
            print("\n✓ Riconciliazione embeddings modelli aggiuntivi...")
            models_to_reconcile = [
                ("image_clip.npy", "CLIP immagini"),
                ("text_clip.npy", "CLIP testo"),
                ("text_minilm.npy", "MiniLM"),
                ("audio_vggish.npy", "VGGish"),
                ("image_vit.npy", "ViT")
            ]
            
            for fname, desc in models_to_reconcile:
                fpath = outdir / fname
                if fpath.exists():
                    arr = np.load(fpath)
                    if arr.shape[0] == len(valid_names) + len(missing_from_current):
                        # Filtra anche questo
                        arr_filtered = arr[indices_to_keep]
                        np.save(fpath, arr_filtered)
                        print(f"  - {fname}: {arr.shape} → {arr_filtered.shape} ({desc})")
                    elif arr.shape[0] == len(valid_names):
                        print(f"  - {fname}: già allineato ({desc})")
                    else:
                        print(f"  ⚠ {fname}: dimensioni non compatibili, skip ({desc})")
            
            print("="*80)
        else:
            print("\n✓ Tutti i file sono presenti, nessuna riconciliazione necessaria")
    else:
        print("Estraggo feature immagini...")
        image_results, image_failures = process_images_ordered(model, preprocess, image_items, device, batch_size=args.batch_size)
        summarize_failures("immagini", image_failures)

        print("Estraggo feature audio (sliding_mean)...")
        audio_results, audio_failures = process_audios_sliding_mean(model, audio_items, device,
                                               sr=args.audio_sr,
                                               window_sec=args.window_sec,
                                               stride_sec=args.stride_sec,
                                               window_batch_size=args.window_batch_size)
        summarize_failures("audio", audio_failures)

        print("Estraggo feature testo...")
        text_results, text_failures = process_texts_ordered(model, text_items, device, batch_size=max(8, args.batch_size))
        summarize_failures("testi", text_failures)

        valid_names = [n for n in common if n in image_results and n in audio_results and n in text_results]
        dropped = [n for n in common if n not in valid_names]
        
        print("\n" + "="*80)
        print("REPORT FINALE ESTRAZIONE")
        print("="*80)
        print(f"Campioni iniziali con tutte e 3 modalità: {len(common)}")
        print(f"Campioni estratti con successo: {len(valid_names)}")
        
        if dropped:
            print(f"\n⚠ Rimossi {len(dropped)} campioni per errori di estrazione:")
            print(f"   Esempi: {dropped[:10]}")
            
            # Dettaglio errori per modalità
            dropped_img = [n for n in dropped if n not in image_results]
            dropped_aud = [n for n in dropped if n not in audio_results]
            dropped_txt = [n for n in dropped if n not in text_results]
            
            if dropped_img:
                print(f"   - {len(dropped_img)} falliti per immagine")
            if dropped_aud:
                print(f"   - {len(dropped_aud)} falliti per audio")
            if dropped_txt:
                print(f"   - {len(dropped_txt)} falliti per testo")
        
        if not valid_names:
            print("\n❌ ERRORE: Nessun embedding estratto con successo. Controlla i messaggi precedenti.")
            sys.exit(1)
        
        print(f"\n✓ Procedendo con {len(valid_names)} campioni validi")
        print("="*80 + "\n")

        images_np = np.stack([image_results[n] for n in valid_names], axis=0).astype(np.float32)
        audios_np = np.stack([audio_results[n] for n in valid_names], axis=0).astype(np.float32)
        texts_np = np.stack([text_results[n] for n in valid_names], axis=0).astype(np.float32)

        # Salva embeddings AudioCLIP
        np.save(outdir / "image_audioclip.npy", images_np)
        print(f"Salvato {outdir/'image_audioclip.npy'} shape={images_np.shape}")
        np.save(outdir / "audio_audioclip.npy", audios_np)
        print(f"Salvato {outdir/'audio_audioclip.npy'} shape={audios_np.shape}")
        np.save(outdir / "text_audioclip.npy", texts_np)
        print(f"Salvato {outdir/'text_audioclip.npy'} shape={texts_np.shape}")
        
        # Salva CSV mapping
        import csv
        with open(outdir / "item_features.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "idx"])
            for i, name in enumerate(valid_names):
                writer.writerow([name, i])
        print(f"Salvato mapping {outdir/'item_features.csv'} N={len(valid_names)}")

    # =============================================================================
    # ESTRAZIONE CON ALTRI MODELLI (stessi campioni valid_names)
    # =============================================================================
    
    print("\n" + "="*80)
    print("INIZIO ESTRAZIONE CON MODELLI AGGIUNTIVI")
    print("="*80)
    
    # CLIP (testo + immagini)
    if (outdir / "image_clip.npy").exists() and (outdir / "text_clip.npy").exists():
        print("\n✓ Feature CLIP già esistenti, skip estrazione")
        image_clip = np.load(outdir / "image_clip.npy")
        text_clip = np.load(outdir / "text_clip.npy")
        print(f"  Caricati: images {image_clip.shape}, texts {text_clip.shape}")
    else:
        try:
            image_clip, text_clip = extract_clip_features(valid_names, img_names, txt_names, device, batch_size=args.batch_size)
            np.save(outdir / "image_clip.npy", image_clip)
            np.save(outdir / "text_clip.npy", text_clip)
            print(f"✓ Salvati embeddings CLIP")
        except Exception as e:
            print(f"❌ Errore CLIP: {e}")
    
    # MiniLM (solo testo)
    if (outdir / "text_minilm.npy").exists():
        print("\n✓ Feature MiniLM già esistenti, skip estrazione")
        text_minilm = np.load(outdir / "text_minilm.npy")
        print(f"  Caricati: {text_minilm.shape}")
    else:
        try:
            text_minilm = extract_minilm_features(valid_names, txt_names, batch_size=args.batch_size)
            np.save(outdir / "text_minilm.npy", text_minilm)
            print(f"✓ Salvati embeddings MiniLM")
        except Exception as e:
            print(f"❌ Errore MiniLM: {e}")
    
    # ViT (solo immagini) - eseguito prima di VGGish per priorità immagini
    if (outdir / "image_vit.npy").exists():
        print("\n✓ Feature ViT già esistenti, skip estrazione")
        image_vit = np.load(outdir / "image_vit.npy")
        print(f"  Caricati: {image_vit.shape}")
    else:
        try:
            image_vit = extract_vit_features(valid_names, img_names, device, batch_size=args.batch_size)
            np.save(outdir / "image_vit.npy", image_vit)
            print(f"✓ Salvati embeddings ViT")
        except Exception as e:
            print(f"❌ Errore ViT: {e}")


    # VGGish (solo audio) - sostituisce Whisper; eseguito dopo ViT
    if (outdir / "audio_vggish.npy").exists():
        print("\n✓ Feature VGGish già esistenti, skip estrazione")
        audio_vggish = np.load(outdir / "audio_vggish.npy")
        print(f"  Caricati: {audio_vggish.shape}")
    else:
        try:
            audio_vggish = extract_vggish_features(
                valid_names, 
                aud_names, 
                vggish_model,
                device=vggish_device,  # Usa GPU se disponibile
                max_workers=8,  # Auto-detect optimal
                batch_inference_size=64  # Batch grande per efficienza GPU
            )
            np.save(outdir / "audio_vggish.npy", audio_vggish)
            print(f"✓ Salvati embeddings VGGish")
        except Exception as e:
            print(f"❌ Errore VGGish: {e}")
            import traceback
            traceback.print_exc()


    # opzionale normalizzazione L2 prima della concatenazione
    if args.l2norm:
        images_np = l2_normalize_rows(images_np)
        audios_np = l2_normalize_rows(audios_np)
        texts_np = l2_normalize_rows(texts_np)

    # concatenazione se richiesta
    if not args.no_concat:
        concatenated = np.concatenate([images_np.astype(np.float32), audios_np.astype(np.float32), texts_np.astype(np.float32)], axis=1)
        np.save(outdir / "concatenated.npy", concatenated)
        print(f"Salvato {outdir/'concatenated.npy'} shape={concatenated.shape}")

    # Salva o aggiorna CSV mapping item_id -> idx se non già fatto
    if not (outdir / "item_features.csv").exists() or not audioclip_files_exist:
        import csv
        with open(outdir / "item_features.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "idx"])
            for i, name in enumerate(valid_names):
                writer.writerow([name, i])
        print(f"Salvato mapping {outdir/'item_features.csv'} N={len(valid_names)}")

    print("\n" + "="*80)
    print("✓ ESTRAZIONE COMPLETATA CON SUCCESSO")
    print("="*80)
    print(f"Files salvati in: {outdir}")
    print(f"\nAudioCLIP:")
    print(f"  - image_audioclip.npy:  {images_np.shape}")
    print(f"  - audio_audioclip.npy:  {audios_np.shape}")
    print(f"  - text_audioclip.npy:   {texts_np.shape}")
    print(f"\nModelli aggiuntivi (se disponibili):")
    
    # Verifica e stampa info sui file salvati
    all_files = {
        "image_clip.npy": "CLIP immagini",
        "text_clip.npy": "CLIP testo",
        "text_minilm.npy": "MiniLM testo",
        "audio_vggish.npy": "VGGish audio",
        "image_vit.npy": "ViT immagini"
    }
    
    for fname, desc in all_files.items():
        fpath = outdir / fname
        if fpath.exists():
            arr = np.load(fpath)
            print(f"  - {fname}: {arr.shape} ({desc})")
    
    if not args.no_concat:
        print(f"\nConcatenato:")
        print(f"  - concatenated.npy: {concatenated.shape}")
    print(f"\nMapping:")
    print(f"  - item_features.csv: {len(valid_names)} righe")
    print("="*80)

if __name__ == "__main__":
    main()