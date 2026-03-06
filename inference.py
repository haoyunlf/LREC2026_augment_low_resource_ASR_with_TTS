import os
import csv
import torch
import torchaudio
from tqdm import tqdm
import pandas as pd
import argparse

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Language code to full name mapping
LANGUAGE_MAPPING = {
    "ug": "uyghur", 
    "sv-SE": "swedish", 
    "ro": "romanian", 
    "cy": "welsh", 
    "et": "estonian", 
    "fy-NL": "frisian", 
    "vi": "vietnanmese",
    "nl": "dutch"
}

# Default synthesis parameters
DEFAULT_SYNTHESIS_PARAMS = {
    "temperature": 0.1,
    "length_penalty": 1.0,
    "repetition_penalty": 10.0,
    "top_k": 10,
    "top_p": 0.3
}


class XTTSInference:
    """XTTS model wrapper for multi-language speech synthesis"""
    
    def __init__(self, language, checkpoint_path, config_path, vocab_path, device=None):
        """
        Initialize XTTS inference engine
        
        Args:
            language (str): Language code (e.g., "fy-NL")
            checkpoint_path (str): Path to model checkpoint
            config_path (str): Path to model config JSON
            vocab_path (str): Path to vocabulary JSON
            device (str): Device to use ("cuda:0", "cpu", etc.)
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.vocab_path = vocab_path
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load XTTS model with specified paths"""
        print(f"Loading XTTS model for {self.language}...")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Config: {self.config_path}")
        print(f"Vocab: {self.vocab_path}")
        
        config = XttsConfig()
        config.load_json(self.config_path)
        
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_path=self.checkpoint_path,
            vocab_path=self.vocab_path,
            use_deepspeed=False,
        )
        self.model.to(self.device)
        print("XTTS model loaded successfully!")
    
    def synthesize_batch(self, metadata_file, output_dir, output_metadata_file=None, 
                        temperature=0.1, length_penalty=1.0, repetition_penalty=10.0, 
                        top_k=10, top_p=0.3):
        """
        Synthesize speech for all entries in metadata file
        
        Args:
            metadata_file (str): Path to input metadata CSV file
            output_dir (str): Directory to save generated audio files
            output_metadata_file (str): Path to save output metadata CSV
            temperature (float): Sampling temperature
            length_penalty (float): Length penalty for generation
            repetition_penalty (float): Repetition penalty
            top_k (int): Top-k sampling
            top_p (float): Top-p sampling
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Read input metadata
        metadata = pd.read_csv(metadata_file, sep='|', header=0, quoting=csv.QUOTE_NONE)
        generated_audio_metadata = []
        
        print(f"Processing {len(metadata)} entries...")
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            try:
                # Extract data
                text = row["text"].replace('"', '')
                speaker_name = row["speaker_name"].replace('"', '')
                speaker_audio_file = row["audio_file"]
                
                # Generate output filename
                file_number = f"{idx+1:05d}"
                utt_id = f'synth_{file_number}_{os.path.basename(speaker_audio_file).split(".")[0]}'
                output_path = os.path.join(output_dir, f"{utt_id}.wav")
                
                # Generate speech
                self._synthesize_single(
                    text=text,
                    speaker_audio_file=speaker_audio_file,
                    output_path=output_path,
                    temperature=temperature,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Store metadata
                generated_audio_metadata.append({
                    "file_name": os.path.basename(output_path),
                    "transcription": text,
                    "speaker_name": speaker_name,
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        # Save output metadata
        if output_metadata_file:
            generated_audio_df = pd.DataFrame(generated_audio_metadata)
            generated_audio_df.to_csv(output_metadata_file, index=False, sep=",")
            print(f"Output metadata saved to: {output_metadata_file}")
        
        print(f"Synthesis complete! Generated {len(generated_audio_metadata)} audio files.")
    
    def _synthesize_single(self, text, speaker_audio_file, output_path, **kwargs):
        """Synthesize speech for a single text-speaker pair"""
        # Generate conditioning latents
        if speaker_audio_file and os.path.exists(speaker_audio_file):
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=speaker_audio_file,
                gpt_cond_len=self.model.config.gpt_cond_len,
                max_ref_length=self.model.config.max_ref_len,
                sound_norm_refs=self.model.config.sound_norm_refs,
            )
        else:
            gpt_cond_latent, speaker_embedding = None, None
        
        # Generate speech
        wav_chunk = self.model.inference(
            text=text,
            language=self.language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **kwargs
        )
        
        # Save audio
        out_wav = torch.tensor(wav_chunk["wav"]).unsqueeze(0).cpu()
        torchaudio.save(output_path, out_wav, sample_rate=self.model.config.audio.sample_rate)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="XTTS Inference for Multi-language Text-to-Speech Synthesis")
    
    # Required arguments
    parser.add_argument("--language", "-l", type=str, required=True,
                        help="Language code (e.g., fy-NL, vi, en)")
    parser.add_argument("--checkpoint", "-c", type=str, 
                        default="checkpoints/frisian/GPT_XTTS_FT-May-25-2025_08+46PM-f1662ee/best_model.pth",
                        help="Path to model checkpoint (.pth file)")
    parser.add_argument("--config", "-cfg", type=str, 
                        default="checkpoints/frisian/GPT_XTTS_FT-May-25-2025_08+46PM-f1662ee/config.json",
                        help="Path to model config (.json file)")
    parser.add_argument("--vocab", "-v", type=str, 
                        default="checkpoints/frisian/XTTS_v2.0_original_model_files/vocab.json",
                        help="Path to vocabulary file (.json file)")
    parser.add_argument("--metadata", "-m", type=str, 
                        default="commonvoice_dataset/small_spk/text_fy-NL_4spk_fy-NL_aug.csv",
                        help="Path to input metadata CSV file")
    parser.add_argument("--output", "-o", type=str, 
                        default="output/wav",
                        help="Output directory for generated audio files (default: output/wav)")
    
    # Optional arguments
    parser.add_argument("--output-metadata", type=str, default=None,
                        help="Path to output metadata CSV file (default: <output_dir>/metadata.csv)")
    
    return parser.parse_args()


def main():
    """Main inference function with command line arguments"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup paths
    metadata_file = args.metadata
    output_dir = args.output
    output_metadata_file = args.output_metadata or os.path.join(output_dir, "metadata.csv")
    
    print(f"Starting XTTS inference for {args.language}")
    print(f"Input metadata: {metadata_file}")
    print(f"Output directory: {output_dir}")
    print(f"Model checkpoint: {args.checkpoint}")
    
    # Initialize inference engine
    inference_engine = XTTSInference(
        language=args.language,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        vocab_path=args.vocab
    )
    
    # Run batch synthesis with default parameters
    inference_engine.synthesize_batch(
        metadata_file=metadata_file,
        output_dir=output_dir,
        output_metadata_file=output_metadata_file,
        **DEFAULT_SYNTHESIS_PARAMS  # Use default synthesis parameters
    )
    
    print(f"✅ Inference completed! Check output at: {output_dir}")


if __name__ == "__main__":
    main()