# XTTSv2 Data Processing and Inference Tools

Tools for processing CommonVoice datasets and performing speech synthesis inference with fine-tuned XTTSv2 models for new languages.

## Pipeline

1. **Data Preparation**: Use `prepare_train_csv.py` to process CommonVoice data
2. **Model Training**: Fine-tune XTTSv2 models on new languages using the training scripts
3. **Inference**: Generate speech using `inference.py` with the trained models

**Note**: Training scripts are based on [XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages).


## Input Requirements

- **CommonVoice original dataset files**: `validated.tsv`, `train.tsv`, `dev.tsv`, `test.tsv`
- **Audio clips folder**: Contains `.mp3` audio files referenced in TSV files
- **CommonVoice structure**: `cv-corpus-XX.X-XXXX-XX-XX/{language_code}/`

## 🚀 Quick Start

### Data Preprocessing
```bash
pip install pandas soundfile
python prepare_train_csv.py
```

**Output Files:**
- `metadata_filtered_validated.csv` - TTS training data
- `metadata_shuffled.csv` - Inference test data
- `metadata_dev.csv` / `metadata_standard_train.csv` - Basic metadata

### TTS Inference
```bash
# Install additional dependencies
pip install TTS torch torchaudio

# Custom model inference
python inference.py \
    --language vi \
    --checkpoint path/to/vietnamese/model.pth \
    --config path/to/vietnamese/config.json \
    --vocab path/to/vietnamese/vocab.json \
    --metadata input_data.csv \
    --output output_directory/
```

**Output Files:**
- Generated audio files in WAV format
- Output metadata CSV with synthesis information

**Output format:** `audio_file|text|speaker_name` (separated by `|`)

## 🔧 Configuration

To modify language processing: change `lang` and `code` parameters in `prepare_train_csv.py` main function.