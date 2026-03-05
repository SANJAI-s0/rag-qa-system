"""
Results Analyzer Module
Analyzes and visualizes evaluation results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    """Analyze and visualize evaluation results"""
    
    def __init__(self, output_dir: str = "outputs/results"):
        self.output_dir = output_dir
        self.results_history = []
        os.makedirs(output_dir, exist_ok=True)
        
    def add_result(self, result: Dict[str, Any]):
        """Add a single result to history"""
        self.results_history.append(result)
        
    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        filepath = f"{self.output_dir}/{filename}"
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "results": self.results_history,
                    "summary": self.generate_summary()
                }, f, indent=2)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.results_history:
            return {}
            
        df = pd.DataFrame(self.results_history)
        
        summary = {
            "total_queries": len(df),
            "average_retrieval_time": float(df['retrieval_time'].mean()) if 'retrieval_time' in df else None,
            "average_generation_time": float(df['generation_time'].mean()) if 'generation_time' in df else None,
            "source_accuracy": float(df['source_accuracy'].mean()) if 'source_accuracy' in df else None,
            "avg_retrieved_docs": float(df['num_retrieved'].mean()) if 'num_retrieved' in df else None
        }
        
        return summary
    
    def plot_results(self):
        """Create visualization plots"""
        if not self.results_history:
            logger.warning("No results to plot")
            return
            
        df = pd.DataFrame(self.results_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Source accuracy by category
        if 'category' in df.columns and 'source_accuracy' in df.columns:
            df.groupby('category')['source_accuracy'].mean().plot(
                kind='bar', ax=axes[0,0], color='skyblue'
            )
            axes[0,0].set_title('Source Accuracy by Category')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].set_ylim([0, 1])
            
        # Plot 2: Response times
        if 'retrieval_time' in df.columns and 'generation_time' in df.columns:
            df[['retrieval_time', 'generation_time']].mean().plot(
                kind='bar', ax=axes[0,1], color=['green', 'orange']
            )
            axes[0,1].set_title('Average Response Times')
            axes[0,1].set_ylabel('Seconds')
            
        # Plot 3: Number of retrieved docs
        if 'num_retrieved' in df.columns:
            df['num_retrieved'].hist(ax=axes[1,0], bins=10, color='purple', alpha=0.7)
            axes[1,0].set_title('Distribution of Retrieved Documents')
            axes[1,0].set_xlabel('Number of Documents')
            
        # Plot 4: Performance over time (if timestamp available)
        if 'timestamp' in df.columns and 'source_accuracy' in df.columns:
            df.sort_values('timestamp')['source_accuracy'].plot(
                ax=axes[1,1], marker='o', color='red'
            )
            axes[1,1].set_title('Source Accuracy Over Time')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].set_ylim([0, 1])
            
        plt.tight_layout()
        plot_path = f"{self.output_dir}/evaluation_plots.png"
        plt.savefig(plot_path)
        plt.show()
        logger.info(f"Plots saved to {plot_path}")