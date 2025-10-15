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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=False, help="Cartella con immagini (.jpg/.png)", default="../ml1m/_images")
    parser.add_argument("--audios", required=False, help="Cartella con audio (.wav)", default="../ml1m/_audios")
    parser.add_argument("--texts", required=False, help="Cartella con testi (.txt)", default="../ml1m/_texts")
    parser.add_argument("--weights", default="AudioCLIP-Full-Training.pt", help="Path al file di pesi AudioCLIP (.pt). Se omesso usa pretrained=True del repo.")
    parser.add_argument("--outdir", default="features_mmrec", help="Cartella output")
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

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # carica modello
    model = load_model(args.weights, device)
    preprocess, res = get_image_preprocess(model)
    print(f"Image preprocess resolution: {res}")

    # lista file e matching per basename
    img_files = sorted([str(p) for p in Path(args.images).glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    audio_files = sorted([str(p) for p in Path(args.audios).glob("*.wav")])
    text_files = sorted([str(p) for p in Path(args.texts).glob("*.txt")])

    print(f"File trovati: {len(img_files)} immagini, {len(audio_files)} audio, {len(text_files)} testi")

    img_names = {Path(p).stem: p for p in img_files}
    aud_names = {Path(p).stem: p for p in audio_files}
    txt_names = {Path(p).stem: p for p in text_files}

    # Analisi dettagliata delle modalità mancanti
    all_names = set(img_names.keys()) | set(aud_names.keys()) | set(txt_names.keys())
    print(f"\nAnalisi basename: {len(all_names)} nomi unici totali")
    
    missing_img = all_names - set(img_names.keys())
    missing_aud = all_names - set(aud_names.keys())
    missing_txt = all_names - set(txt_names.keys())
    
    if missing_img:
        print(f"  ⚠ {len(missing_img)} nomi senza immagine. Esempi: {sorted(list(missing_img))[:10]}")
    if missing_aud:
        print(f"  ⚠ {len(missing_aud)} nomi senza audio. Esempi: {sorted(list(missing_aud))[:10]}")
    if missing_txt:
        print(f"  ⚠ {len(missing_txt)} nomi senza testo. Esempi: {sorted(list(missing_txt))[:10]}")

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

    np.save(outdir / "images.npy", images_np)
    print(f"Salvato {outdir/'images.npy'} shape={images_np.shape}")
    np.save(outdir / "audios.npy", audios_np)
    print(f"Salvato {outdir/'audios.npy'} shape={audios_np.shape}")
    np.save(outdir / "texts.npy", texts_np)
    print(f"Salvato {outdir/'texts.npy'} shape={texts_np.shape}")

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

    # salva CSV mapping item_id -> idx (non salviamo names.npy come richiesto)
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
    print(f"  - images.npy:  {images_np.shape}")
    print(f"  - audios.npy:  {audios_np.shape}")
    print(f"  - texts.npy:   {texts_np.shape}")
    if not args.no_concat:
        print(f"  - concatenated.npy: {concatenated.shape}")
    print(f"  - item_features.csv: {len(valid_names)} righe")
    print("="*80)

if __name__ == "__main__":
    main()