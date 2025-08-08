#!/usr/bin/env python3
"""
Comprehensive LLM Brain Analysis Framework
Extended version with 50+ models, 20+ datasets, and advanced cognitive tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    AutoModelForSequenceClassification, pipeline
)
import torch
import torch.nn.functional as F
from datasets import load_dataset
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import gc
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings('ignore')

class ExtendedLLMBrainAnalyzer:
    """
    Comprehensive LLM Brain Analysis with 50+ models and 20+ cognitive tests
    """
    
    def __init__(self, device='auto', max_models_parallel=2):
        # Enhanced device detection
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        print(f"🧠 Extended LLM Brain Analyzer initialized")
        print(f"📱 Device: {self.device}")
        print(f"🔄 Max parallel models: {max_models_parallel}")
        
        self.max_models_parallel = max_models_parallel
        self.models = {}
        self.tokenizers = {}
        self.results = {}
        self.failed_models = []
        self.successful_models = []
        
        # Extended brain regions with more specific mappings
        self.brain_regions = {
            'visual_cortex': ['pattern_recognition', 'visual_processing', 'feature_extraction'],
            'temporal_lobe': ['language_processing', 'semantic_memory', 'word_recognition'],
            'parietal_lobe': ['spatial_reasoning', 'numerical_processing', 'attention_control'],
            'prefrontal_cortex': ['executive_function', 'reasoning', 'planning', 'decision_making'],
            'hippocampus': ['episodic_memory', 'sequence_learning', 'context_processing'],
            'broca_area': ['language_production', 'syntax_processing'],
            'wernicke_area': ['language_comprehension', 'semantic_processing'],
            'anterior_cingulate': ['conflict_monitoring', 'error_detection'],
            'cerebellum': ['procedural_learning', 'motor_coordination', 'prediction'],
            'amygdala': ['emotion_processing', 'threat_detection', 'valence_assessment']
        }
        
        # Comprehensive model list (50+ models)
        self.llm_models = {
            # BERT Family
            'bert_family': [
                'bert-base-uncased',
                'bert-large-uncased',
                'distilbert-base-uncased',
                'roberta-base',
                'roberta-large',
                'albert-base-v2',
                'albert-large-v2',
                'microsoft/deberta-base',
                'microsoft/deberta-large',
                'distilroberta-base',
            ],
            
            # GPT Family  
            'gpt_family': [
                'gpt2',
                'gpt2-medium',
                'gpt2-large',
                'microsoft/DialoGPT-small',
                'microsoft/DialoGPT-medium',
               'microsoft/DialoGPT-large',
                'EleutherAI/gpt-neo-125M',
                'EleutherAI/gpt-neo-1.3B',
                'EleutherAI/gpt-neo-2.7B',
               #
               #  'EleutherAI/gpt-j-6b',
            ],
            
            # Specialized Models
            'specialized': [
                'sentence-transformers/all-MiniLM-L6-v2',
                'sentence-transformers/all-mpnet-base-v2',
                'facebook/dpr-ctx_encoder-single-nq-base',
                'microsoft/codebert-base',
                'huggingface/CodeBERTa-small-v1',
                'allenai/longformer-base-4096',
                'google/electra-base-discriminator',
                'google/electra-large-discriminator',
            ],
            
            # OPT Family
            'opt_family': [
                'facebook/opt-125m',
                'facebook/opt-350m',
                'facebook/opt-1.3b',
                'facebook/opt-2.7b',
            ],
            
            # BLOOM Family  
            'bloom_family': [
                'bigscience/bloom-560m',
                'bigscience/bloom-1b1',
                'bigscience/bloom-1b7',
                'bigscience/bloom-3b',
            ],
            
            # Multilingual Models
            'multilingual': [
                'xlm-roberta-base',
                'xlm-roberta-large', 
                'microsoft/mdeberta-v3-base',
                'google/muril-base-cased',
            ],
            
            # Domain-Specific
            'domain_specific': [
                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                'allenai/scibert_scivocab_uncased',
                'nlpaueb/legal-bert-base-uncased',
                'ProsusAI/finbert',
                'climatebert/distilroberta-base-climate-f',
            ]
        }
        
        # Flatten model list
        self.all_models = []
        for family, models in self.llm_models.items():
            self.all_models.extend(models)
        
        # Extended dataset collection (20+ datasets)
        self.cognitive_datasets = {
            # Language Understanding
            'language_understanding': {
                'sentiment_analysis': 'glue/sst2',
                'text_classification': 'ag_news', 
                'natural_language_inference': 'glue/mnli',
                'paraphrase_detection': 'glue/mrpc',
                'linguistic_acceptability': 'glue/cola',
            },
            
            # Reasoning & Logic
            'reasoning_logic': {
                'commonsense_reasoning': 'winogrande/winogrande_debiased',
                'causal_reasoning': 'super_glue/copa',
                'textual_entailment': 'super_glue/rte',
                'reading_comprehension': 'squad',
                'logical_reasoning': 'super_glue/cb',
            },
            
            # Memory & Context
            'memory_context': {
                'long_context': 'narrativeqa',
                'short_memory': 'babi_qa',
                'episodic_memory': 'squad',
                'semantic_memory': 'conceptnet',
            },
            
            # Mathematical & Numerical
            'numerical_spatial': {
                'mathematical_reasoning': 'math_qa',
                'numerical_understanding': 'drop',
                'arithmetic': 'math_qa',
                'word_problems': 'math_qa',
            },
            
            # Creative & Generation
            'creative_generation': {
                'story_completion': 'writingprompts',
                'dialogue_generation': 'daily_dialog',
                'creative_writing': 'poem_sentiment',
            },
            
            # Specialized Cognitive Tasks
            'specialized_cognition': {
                'emotion_recognition': 'emotion',
                'irony_detection': 'tweet_irony',
                'hate_speech_detection': 'hate_speech18',
                'fact_checking': 'fever',
                'question_answering': 'natural_questions',
            }
        }
        
        # Flatten datasets
        self.all_datasets = {}
        for category, datasets in self.cognitive_datasets.items():
            self.all_datasets.update(datasets)

    def get_model_architecture_type(self, model_name: str) -> str:
        """Determine model architecture type"""
        if any(x in model_name.lower() for x in ['t5', 'flan', 'bart', 'pegasus']):
            return 'encoder_decoder'
        elif any(x in model_name.lower() for x in ['gpt', 'opt', 'bloom', 'neo']):
            return 'decoder_only'
        else:
            return 'encoder_only'

    def load_model_safe(self, model_name: str) -> bool:
        """Safely load model with architecture-specific handling"""
        try:
            print(f"🔄 Loading {model_name}...")
            
            arch_type = self.get_model_architecture_type(model_name)
            
            # Load tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            
            if self.tokenizers[model_name].pad_token is None:
                self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token
            
            # Load model based on architecture
            if arch_type == 'encoder_decoder':
                print(f"  ⚠️  Skipping {model_name}: Encoder-decoder architecture not supported in this version")
                return False
                
            elif arch_type == 'decoder_only':
                # For generative models
                try:
                    self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32 if self.device in ['cpu', 'mps'] else torch.float16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        attn_implementation="eager"  # Fix attention warnings
                    ).to(self.device)
                except:
                    # Fallback to AutoModel
                    self.models[model_name] = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        attn_implementation="eager"
                    ).to(self.device)
                    
            else:  # encoder_only
                self.models[model_name] = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32 if self.device in ['cpu', 'mps'] else torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation="eager"  # Fix attention warnings
                ).to(self.device)
            
            print(f"  ✅ Successfully loaded {model_name}")
            self.successful_models.append(model_name)
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load {model_name}: {str(e)}")
            self.failed_models.append(model_name)
            return False

    def extract_brain_activations(self, model_name: str, text: str, max_length: int = 256) -> Dict:
        """Extract detailed brain-like activation patterns"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            activations = {}
            
            with torch.no_grad():
                # Get model outputs
                outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
                
                if not hasattr(outputs, 'hidden_states') or not outputs.hidden_states:
                    return {}
                
                hidden_states = outputs.hidden_states
                num_layers = len(hidden_states)
                
                # Advanced brain region mapping
                layer_thirds = num_layers // 3
                
                # Visual Cortex (Early sensory processing)
                early_layers = hidden_states[:max(1, layer_thirds//2)]
                if early_layers:
                    activations['visual_cortex'] = torch.mean(torch.stack([
                        torch.mean(layer.abs()) for layer in early_layers
                    ])).item()
                
                # Temporal Lobe (Language and semantic processing)
                temporal_start = max(1, layer_thirds//2)
                temporal_end = max(2, layer_thirds)
                temporal_layers = hidden_states[temporal_start:temporal_end]
                if temporal_layers:
                    activations['temporal_lobe'] = torch.mean(torch.stack([
                        torch.mean(layer.abs()) for layer in temporal_layers
                    ])).item()
                
                # Broca's Area (Language production - middle layers)
                broca_start = layer_thirds
                broca_end = min(num_layers-1, layer_thirds + layer_thirds//2)
                broca_layers = hidden_states[broca_start:broca_end]
                if broca_layers:
                    activations['broca_area'] = torch.mean(torch.stack([
                        torch.mean(layer.abs()) for layer in broca_layers
                    ])).item()
                
                # Parietal Lobe (Attention and spatial processing)
                parietal_start = max(broca_end, 2*layer_thirds//3)
                parietal_end = max(parietal_start+1, 2*layer_thirds)
                parietal_layers = hidden_states[parietal_start:parietal_end]
                if parietal_layers:
                    activations['parietal_lobe'] = torch.mean(torch.stack([
                        torch.mean(layer.abs()) for layer in parietal_layers
                    ])).item()
                
                # Prefrontal Cortex (Executive function, reasoning)
                prefrontal_start = max(parietal_end, 2*layer_thirds)
                prefrontal_layers = hidden_states[prefrontal_start:-1] if prefrontal_start < num_layers-1 else [hidden_states[-2]]
                if prefrontal_layers:
                    activations['prefrontal_cortex'] = torch.mean(torch.stack([
                        torch.mean(layer.abs()) for layer in prefrontal_layers
                    ])).item()
                
                # Wernicke's Area (Language comprehension - specific attention patterns)
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    # Use middle attention layers for comprehension
                    mid_attention = outputs.attentions[len(outputs.attentions)//2:2*len(outputs.attentions)//3]
                    if mid_attention:
                        activations['wernicke_area'] = torch.mean(torch.stack([
                            torch.mean(attn) for attn in mid_attention
                        ])).item()
                
                # Hippocampus (Memory and sequence processing)
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    # Use late attention layers for memory
                    late_attention = outputs.attentions[-3:] if len(outputs.attentions) > 3 else outputs.attentions
                    activations['hippocampus'] = torch.mean(torch.stack([
                        torch.mean(attn) for attn in late_attention
                    ])).item()
                
                # Anterior Cingulate (Error monitoring - attention variance)
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    attention_variance = torch.var(torch.stack([
                        torch.mean(attn) for attn in outputs.attentions
                    ])).item()
                    activations['anterior_cingulate'] = attention_variance
                
                # Amygdala (Emotional processing - activation intensity)
                final_layer = hidden_states[-1]
                emotional_intensity = torch.std(final_layer).item()
                activations['amygdala'] = emotional_intensity
                
                # Cerebellum (Prediction and coordination - gradient smoothness)
                if len(hidden_states) > 2:
                    layer_diffs = []
                    for i in range(1, len(hidden_states)):
                        diff = torch.mean((hidden_states[i] - hidden_states[i-1]).abs()).item()
                        layer_diffs.append(diff)
                    activations['cerebellum'] = np.mean(layer_diffs) if layer_diffs else 0.0
                
            return activations
            
        except Exception as e:
            print(f"    ⚠️  Error extracting activations: {e}")
            return {}

    def load_dataset_safe(self, dataset_name: str, num_samples: int = 50) -> List[Dict]:
        """Safely load and sample from datasets"""
        try:
            print(f"📊 Loading dataset: {dataset_name}")
            
            # Special handling for different dataset configurations
            if dataset_name == 'winogrande/winogrande_debiased':
                dataset = load_dataset('winogrande', 'winogrande_debiased', split='train')
            elif dataset_name == 'narrativeqa':
                dataset = load_dataset('narrativeqa', split='train')
            elif dataset_name == 'babi_qa':
                dataset = load_dataset('facebook/babi_qa', 'en-valid-10k-1', split='train')
            elif dataset_name == 'conceptnet':
                # Use a simpler alternative
                dataset = load_dataset('glue', 'wnli', split='train')
                print(f"  📝 Using WNLI as conceptnet alternative")
            elif dataset_name == 'writingprompts':
                dataset = load_dataset('writingprompts', split='train')
            elif dataset_name == 'daily_dialog':
                dataset = load_dataset('daily_dialog', split='train')
            elif dataset_name == 'poem_sentiment':
                # Use emotion dataset as alternative
                dataset = load_dataset('emotion', split='train')
                print(f"  📝 Using emotion dataset as poem_sentiment alternative")
            elif dataset_name == 'tweet_irony':
                dataset = load_dataset('tweet_eval', 'irony', split='train')
            elif dataset_name == 'hate_speech18':
                dataset = load_dataset('hate_speech18', split='train')
            elif dataset_name == 'fever':
                dataset = load_dataset('fever', 'v1.0', split='train')
            elif dataset_name == 'natural_questions':
                dataset = load_dataset('natural_questions', split='train')
            elif '/' in dataset_name:
                parts = dataset_name.split('/')
                dataset = load_dataset(parts[0], parts[1], split='train')
            else:
                dataset = load_dataset(dataset_name, split='train')
            
            # Sample data
            dataset_size = len(dataset)
            sample_size = min(num_samples, dataset_size)
            samples = dataset.shuffle(seed=42).select(range(sample_size))
            
            print(f"  ✅ Loaded {sample_size} samples from {dataset_name}")
            return list(samples)
            
        except Exception as e:
            print(f"  ❌ Failed to load {dataset_name}: {e}")
            return []

    def extract_text_from_sample(self, sample: Dict, task_type: str) -> str:
        """Smart text extraction based on task type and sample structure"""
        
        # Task-specific text extraction
        text_extractors = {
            'sentiment_analysis': lambda s: s.get('sentence', s.get('text', '')),
            'text_classification': lambda s: s.get('text', ''),
            'natural_language_inference': lambda s: f"Premise: {s.get('premise', '')} Hypothesis: {s.get('hypothesis', '')}",
            'reading_comprehension': lambda s: f"Context: {s.get('context', '')[:200]} Question: {s.get('question', '')}",
            'mathematical_reasoning': lambda s: s.get('Problem', s.get('question', '')),
            'commonsense_reasoning': lambda s: f"{s.get('sentence', '')} Option 1: {s.get('option1', '')} Option 2: {s.get('option2', '')}",
            'emotion_recognition': lambda s: s.get('text', ''),
        }
        
        # Try specific extractor first
        if task_type in text_extractors:
            text = text_extractors[task_type](sample)
            if text.strip():
                return text.strip()
        
        # Fallback: find any text field
        text_fields = ['text', 'sentence', 'premise', 'hypothesis', 'question', 'context', 'Problem']
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                return sample[field][:500]  # Limit length
        
        # Last resort: convert to string
        return str(sample)[:200]

    def test_cognitive_task(self, task_name: str, dataset_name: str, num_samples: int = 30) -> Dict:
        """Test multiple models on a cognitive task"""
        
        print(f"\n🧠 === COGNITIVE TEST: {task_name.upper()} ===")
        
        # Load dataset
        samples = self.load_dataset_safe(dataset_name, num_samples)
        if not samples:
            return {}
        
        task_results = {}
        
        # Test each model (limit to prevent memory issues)
        tested_models = 0
        max_models_per_task = 100
        
        for model_name in self.all_models:
            if tested_models >= max_models_per_task:
                break
                
            print(f"  🤖 Testing {model_name}...")
            
            if not self.load_model_safe(model_name):
                continue
            
            model_brain_activity = []
            
            # Test on samples
            for i, sample in enumerate(samples[:10]):  # Limit samples per model
                try:
                    text = self.extract_text_from_sample(sample, task_name)
                    if len(text.strip()) < 5:  # Skip very short texts
                        continue
                        
                    brain_activity = self.extract_brain_activations(model_name, text)
                    if brain_activity:
                        model_brain_activity.append(brain_activity)
                        
                except Exception as e:
                    print(f"    ⚠️  Sample {i} failed: {e}")
                    continue
            
            if model_brain_activity:
                # Average brain activity across samples
                avg_brain_activity = {}
                for region in self.brain_regions.keys():
                    activities = [ba.get(region, 0.0) for ba in model_brain_activity]
                    avg_brain_activity[region] = np.mean(activities) if activities else 0.0
                
                task_results[model_name] = avg_brain_activity
                tested_models += 1
                print(f"    ✅ Completed {model_name}")
            
            # Clean up memory
            self.cleanup_model(model_name)
        
        return task_results

    def cleanup_model(self, model_name: str):
        """Clean up model from memory"""
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.tokenizers:
            del self.tokenizers[model_name]
        
        gc.collect()
        if self.device == 'mps':
            torch.mps.empty_cache()
        elif self.device == 'cuda':
            torch.cuda.empty_cache()

    def run_comprehensive_analysis(self, output_dir: str = 'extended_llm_brain_analysis'):
        """Run comprehensive cognitive analysis"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("🧠 === EXTENDED LLM BRAIN ANALYSIS FRAMEWORK ===")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Total models to test: {len(self.all_models)}")
        print(f"📊 Total cognitive tasks: {len(self.all_datasets)}")
        
        all_results = {}
        successful_tasks = 0
        
        # Test each cognitive task
        for task_name, dataset_name in self.all_datasets.items():
            try:
                task_results = self.test_cognitive_task(task_name, dataset_name)
                
                if task_results:
                    all_results[task_name] = task_results
                    successful_tasks += 1
                    
                    # Save individual task results
                    with open(f'{output_dir}/{task_name}_brain_activity.json', 'w') as f:
                        json.dump({
                            model: {region: float(activity) for region, activity in activities.items()}
                            for model, activities in task_results.items()
                        }, f, indent=2)
                
            except Exception as e:
                print(f"❌ Task {task_name} failed: {e}")
                continue
        
        # Generate comprehensive analysis
        if all_results:
            print(f"\n🎯 Successfully completed {successful_tasks} cognitive tasks!")
            
            # Create advanced visualizations
            self.create_advanced_visualizations(all_results, output_dir)
            
            # Analyze cognitive patterns
            cognitive_analysis = self.analyze_cognitive_patterns(all_results)
            
            # Create model comparison
            model_comparison = self.compare_model_cognitive_abilities(all_results)
            
            # Save comprehensive results
            final_results = {
                'brain_activity_by_task': all_results,
                'cognitive_analysis': cognitive_analysis,
                'model_comparison': model_comparison,
                'successful_models': self.successful_models,
                'failed_models': self.failed_models,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f'{output_dir}/comprehensive_results.json', 'w') as f:
                json.dump(final_results, f, indent=2)
            
            # Print executive summary
            self.print_executive_summary(cognitive_analysis, model_comparison)
            
        else:
            print("❌ No successful results to analyze")
        
        return all_results

    def create_advanced_visualizations(self, all_results: Dict, output_dir: str):
        """Create multiple advanced visualizations"""
        
        print("\n🎨 Creating advanced visualizations...")
        
        # 1. Comprehensive brain activity heatmap
        self.create_mega_heatmap(all_results, f'{output_dir}/mega_brain_heatmap.html')
        
        # 2. Cognitive radar charts
        self.create_cognitive_radar_charts(all_results, f'{output_dir}/cognitive_radar.html')
        
        # 3. Model similarity analysis
        self.create_model_similarity_plot(all_results, f'{output_dir}/model_similarity.html')
        
        # 4. Brain region specialization chart
        self.create_specialization_chart(all_results, f'{output_dir}/brain_specialization.html')
        
        print("  ✅ All visualizations created!")

    def create_mega_heatmap(self, all_results: Dict, save_path: str):
        """Create comprehensive heatmap of all results"""
        
        # Prepare data
        heatmap_data = []
        for task, task_results in all_results.items():
            for model, brain_activity in task_results.items():
                for region, activity in brain_activity.items():
                    heatmap_data.append({
                        'Task': task.replace('_', ' ').title(),
                        'Model': model.split('/')[-1],
                        'Brain_Region': region.replace('_', ' ').title(),
                        'Activity': activity
                    })
        
        df = pd.DataFrame(heatmap_data)
        
        if len(df) > 0:
            # Create pivot table
            pivot = df.pivot_table(
                index=['Task'], 
                columns=['Brain_Region'], 
                values='Activity',
                aggfunc='mean'
            )
            
            # Create interactive heatmap
            fig = px.imshow(
                pivot.values,
                x=pivot.columns,
                y=pivot.index,
                color_continuous_scale='Viridis',
                title='Comprehensive LLM Brain Activity Patterns'
            )
            
            fig.update_layout(
                xaxis_title="Brain Regions",
                yaxis_title="Cognitive Tasks",
                height=600
            )
            
            fig.write_html(save_path)

    def create_cognitive_radar_charts(self, all_results: Dict, save_path: str):
        """Create radar charts showing cognitive profiles"""
        
        # Calculate average brain activity per task
        task_profiles = {}
        
        for task, task_results in all_results.items():
            brain_regions = list(self.brain_regions.keys())
            avg_activity = {}
            
            for region in brain_regions:
                activities = [result.get(region, 0.0) for result in task_results.values()]
                avg_activity[region] = np.mean(activities) if activities else 0.0
            
            task_profiles[task] = avg_activity
        
        # Create subplots
        num_tasks = len(task_profiles)
        cols = min(3, num_tasks)
        rows = (num_tasks + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'polar'}] * cols for _ in range(rows)],
            subplot_titles=list(task_profiles.keys())
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (task, profile) in enumerate(task_profiles.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            regions = list(profile.keys())
            values = list(profile.values())
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=[r.replace('_', ' ').title() for r in regions],
                    fill='toself',
                    name=task,
                    line_color=colors[i % len(colors)]
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Cognitive Task Brain Activation Profiles",
            height=300 * rows
        )
        
        fig.write_html(save_path)

    def create_model_similarity_plot(self, all_results: Dict, save_path: str):
        """Create model similarity analysis based on brain patterns"""
        
        # Collect all model brain patterns
        model_patterns = {}
        
        for task, task_results in all_results.items():
            for model, brain_activity in task_results.items():
                if model not in model_patterns:
                    model_patterns[model] = {}
                
                for region, activity in brain_activity.items():
                    key = f"{task}_{region}"
                    model_patterns[model][key] = activity
        
        # Create feature matrix
        all_features = set()
        for patterns in model_patterns.values():
            all_features.update(patterns.keys())
        
        feature_matrix = []
        model_names = []
        
        for model, patterns in model_patterns.items():
            features = [patterns.get(feature, 0.0) for feature in sorted(all_features)]
            feature_matrix.append(features)
            model_names.append(model.split('/')[-1])
        
        # Compute similarity and PCA
        if len(feature_matrix) > 1:
            similarity_matrix = cosine_similarity(feature_matrix)
            
            # PCA for 2D visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(feature_matrix)
            
            # Create scatter plot with fixed color handling
            color_values = [i for i in range(len(model_names))]  # Explicit list creation
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                mode='markers+text',
                text=model_names,
                textposition="top center",
                marker=dict(
                    size=10, 
                    color=color_values,  # Use explicit list
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Models'
            ))
            
            fig.update_layout(
                title='LLM Cognitive Pattern Similarity (PCA)',
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                height=600
            )
            
            fig.write_html(save_path)
            print(f"  ✅ Model similarity plot saved to {save_path}")
        else:
            print(f"  ⚠️  Not enough data for similarity plot")

    def create_specialization_chart(self, all_results: Dict, save_path: str):
        """Show which brain regions specialize in which tasks"""
        
        # Calculate specialization scores
        specialization_data = []
        
        for region in self.brain_regions.keys():
            region_activities = {}
            
            for task, task_results in all_results.items():
                activities = [result.get(region, 0.0) for result in task_results.values()]
                region_activities[task] = np.mean(activities) if activities else 0.0
            
            # Find most specialized tasks for this region
            if region_activities:
                max_task = max(region_activities, key=region_activities.get)
                specialization_score = region_activities[max_task]
                
                specialization_data.append({
                    'Brain_Region': region.replace('_', ' ').title(),
                    'Specialized_Task': max_task.replace('_', ' ').title(),
                    'Specialization_Score': specialization_score
                })
        
        df = pd.DataFrame(specialization_data)
        
        if len(df) > 0:
            fig = px.bar(
                df,
                x='Brain_Region',
                y='Specialization_Score',
                color='Specialized_Task',
                title='Brain Region Task Specialization',
                height=500
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            fig.write_html(save_path)

    def analyze_cognitive_patterns(self, all_results: Dict) -> Dict:
        """Comprehensive cognitive pattern analysis"""
        
        analysis = {
            'task_brain_mapping': {},
            'model_cognitive_profiles': {},
            'brain_region_specialization': {},
            'cognitive_efficiency_scores': {}
        }
        
        # Analyze task-brain mapping
        for task, task_results in all_results.items():
            # Average brain activity across all models for this task
            avg_brain_activity = {}
            for region in self.brain_regions.keys():
                activities = [result.get(region, 0.0) for result in task_results.values()]
                avg_brain_activity[region] = np.mean(activities) if activities else 0.0
            
            # Find dominant brain regions
            sorted_regions = sorted(avg_brain_activity.items(), key=lambda x: x[1], reverse=True)
            analysis['task_brain_mapping'][task] = {
                'dominant_regions': sorted_regions[:3],
                'brain_pattern': self.interpret_brain_pattern(sorted_regions)
            }
        
        # Analyze model cognitive profiles
        for model in self.successful_models:
            model_profile = {}
            for task, task_results in all_results.items():
                if model in task_results:
                    # Calculate cognitive efficiency for this model on this task
                    brain_activity = task_results[model]
                    # Use prefrontal cortex activity as reasoning indicator
                    reasoning_score = brain_activity.get('prefrontal_cortex', 0.0)
                    # Use hippocampus as memory indicator
                    memory_score = brain_activity.get('hippocampus', 0.0)
                    # Use temporal lobe as language indicator
                    language_score = brain_activity.get('temporal_lobe', 0.0)
                    
                    model_profile[task] = {
                        'reasoning': reasoning_score,
                        'memory': memory_score,
                        'language': language_score,
                        'overall': np.mean([reasoning_score, memory_score, language_score])
                    }
            
            analysis['model_cognitive_profiles'][model] = model_profile
        
        return analysis

    def interpret_brain_pattern(self, sorted_regions: List[Tuple[str, float]]) -> str:
        """Interpret brain activation patterns"""
        
        top_region = sorted_regions[0][0]
        
        interpretations = {
            'prefrontal_cortex': 'Executive reasoning and planning dominant',
            'temporal_lobe': 'Language processing and semantic memory active',
            'parietal_lobe': 'Attention and spatial-numerical processing engaged',
            'hippocampus': 'Memory formation and contextual learning active',
            'visual_cortex': 'Pattern recognition and feature extraction',
            'broca_area': 'Language production and syntax processing',
            'wernicke_area': 'Language comprehension and meaning extraction',
            'anterior_cingulate': 'Conflict monitoring and error detection',
            'amygdala': 'Emotional processing and valence assessment',
            'cerebellum': 'Predictive processing and sequence coordination'
        }
        
        return interpretations.get(top_region, 'Mixed cognitive processing')

    def compare_model_cognitive_abilities(self, all_results: Dict) -> Dict:
        """Compare cognitive abilities across models"""
        
        comparison = {
            'best_reasoning_models': [],
            'best_memory_models': [],
            'best_language_models': [],
            'most_brain_like_models': [],
            'model_rankings': {}
        }
        
        # Calculate model scores
        model_scores = {}
        
        for model in self.successful_models:
            scores = {
                'reasoning': [],
                'memory': [],
                'language': [],
                'overall_brain_activity': []
            }
            
            for task, task_results in all_results.items():
                if model in task_results:
                    brain_activity = task_results[model]
                    
                    scores['reasoning'].append(brain_activity.get('prefrontal_cortex', 0.0))
                    scores['memory'].append(brain_activity.get('hippocampus', 0.0))
                    scores['language'].append(brain_activity.get('temporal_lobe', 0.0))
                    scores['overall_brain_activity'].append(np.mean(list(brain_activity.values())))
            
            # Average scores
            model_scores[model] = {
                'reasoning': np.mean(scores['reasoning']) if scores['reasoning'] else 0.0,
                'memory': np.mean(scores['memory']) if scores['memory'] else 0.0,
                'language': np.mean(scores['language']) if scores['language'] else 0.0,
                'brain_like': np.mean(scores['overall_brain_activity']) if scores['overall_brain_activity'] else 0.0
            }
        
        # Find best models in each category
        if model_scores:
            comparison['best_reasoning_models'] = sorted(
                model_scores.items(), key=lambda x: x[1]['reasoning'], reverse=True
            )[:5]
            
            comparison['best_memory_models'] = sorted(
                model_scores.items(), key=lambda x: x[1]['memory'], reverse=True
            )[:5]
            
            comparison['best_language_models'] = sorted(
                model_scores.items(), key=lambda x: x[1]['language'], reverse=True
            )[:5]
            
            comparison['most_brain_like_models'] = sorted(
                model_scores.items(), key=lambda x: x[1]['brain_like'], reverse=True
            )[:5]
        
        comparison['model_rankings'] = model_scores
        
        return comparison

    def print_executive_summary(self, cognitive_analysis: Dict, model_comparison: Dict):
        """Print executive summary of findings"""
        
        print(f"\n{'='*60}")
        print("🧠 EXECUTIVE SUMMARY: LLM BRAIN ANALYSIS")
        print(f"{'='*60}")
        
        # Task-Brain Region Mapping
        print("\n🎯 COGNITIVE TASK → BRAIN REGION MAPPING:")
        for task, analysis in cognitive_analysis['task_brain_mapping'].items():
            dominant = analysis['dominant_regions'][0]
            print(f"  {task.replace('_', ' ').title():25} → {dominant[0].replace('_', ' ').title()} ({dominant[1]:.3f})")
        
        # Best Models
        print(f"\n🏆 TOP PERFORMING MODELS:")
        
        if model_comparison['best_reasoning_models']:
            print(f"  🧮 Best Reasoning: {model_comparison['best_reasoning_models'][0][0].split('/')[-1]}")
        
        if model_comparison['best_memory_models']:
            print(f"  🧠 Best Memory: {model_comparison['best_memory_models'][0][0].split('/')[-1]}")
        
        if model_comparison['best_language_models']:
            print(f"  💬 Best Language: {model_comparison['best_language_models'][0][0].split('/')[-1]}")
        
        if model_comparison['most_brain_like_models']:
            print(f"  🔬 Most Brain-like: {model_comparison['most_brain_like_models'][0][0].split('/')[-1]}")
        
        print(f"\n📊 ANALYSIS COMPLETE!")
        print(f"✅ Successfully tested models: {len(self.successful_models)}")
        print(f"❌ Failed models: {len(self.failed_models)}")

def main():
    """Run the extended analysis"""
    analyzer = ExtendedLLMBrainAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print("\n🎉 Extended LLM Brain Analysis Complete!")
    print("📁 Check the 'extended_llm_brain_analysis' folder for detailed results")
    print("🌐 Open the HTML files for interactive visualizations")

if __name__ == "__main__":
    main()