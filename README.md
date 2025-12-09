

## Embedding Extraction

This repository includes a script `extract_embeddings.py` for extracting multimodal embeddings from custom datasets using AudioCLIP and other models.

### Requirements

- Python >= 3.7
- Install dependencies: `pip install -r requirements.txt`
- Additional models: CLIP, MiniLM, VGGish, ViT (installed via transformers and sentence-transformers)

### Usage

Run the script with the following command:

```bash
python extract_embeddings.py --images /path/to/images --audios /path/to/audios --texts /path/to/texts --outdir output_directory
```

### Parameters

- `--images`: Path to the folder containing images (.jpg, .jpeg, .png)
- `--audios`: Path to the folder containing audio files (.wav)
- `--texts`: Path to the folder containing text files (.txt)
- `--weights`: Path to AudioCLIP weights file (.pt). Defaults to pretrained if not provided.
- `--outdir`: Output directory for embeddings. Defaults to `lastfm_features`.
- `--device`: Device to use (cuda or cpu). Defaults to cuda:1.
- `--batch-size`: Batch size for processing. Defaults to 512.
- `--window-batch-size`: Batch size for audio windows. Defaults to 512.
- `--audio-sr`: Audio sample rate. Defaults to 44100.
- `--window-sec`: Window length in seconds for audio. Defaults to 2.0.
- `--stride-sec`: Stride in seconds for audio. Defaults to 1.0.
- `--no-concat`: Do not create concatenated.npy file.
- `--l2norm`: Apply L2 normalization before concatenation.

### Output

The script generates the following files in the output directory:

- `image_audioclip.npy`, `audio_audioclip.npy`, `text_audioclip.npy`: AudioCLIP embeddings
- `image_clip.npy`, `text_clip.npy`: CLIP embeddings
- `text_minilm.npy`: MiniLM embeddings
- `audio_vggish.npy`: VGGish embeddings
- `image_vit.npy`: ViT embeddings
- `item_features.csv`: Mapping of item IDs to indices
- `concatenated.npy`: Concatenated embeddings (if not disabled)

### Notes

- The script validates audio files and skips corrupted ones.
- Embeddings are extracted only for items present in all three modalities (image, audio, text).
- If embeddings already exist, the script loads them and reconciles with current data.
