"""
Script to analyze debate statistics across all pipeline outputs.
"""
import json
from pathlib import Path
from collections import Counter

def analyze_debate_stats():
    stats = Counter()
    stats['total'] = 0
    stats['no_debate'] = 0
    stats['round1'] = 0
    stats['round2'] = 0
    stats['redline'] = 0
    stats['error'] = 0
    
    # Track consensus rates
    consensus_rates = []
    
    voting_dir = Path('debug_outputs/voting_engine')
    final_files = list(voting_dir.glob('*_03_final_json.json'))
    
    for f in final_files:
        try:
            with open(f, encoding='utf-8') as fp:
                d = json.load(fp)
            stats['total'] += 1
            
            # Check for debate_rounds in tier_4 model_meta
            meta = d.get('tier_4_metadata_and_future', {}).get('model_meta', {})
            debate_rounds = meta.get('debate_rounds', 0)
            consensus_rate = meta.get('consensus_rate', None)
            
            if consensus_rate:
                consensus_rates.append(consensus_rate)
            
            if debate_rounds == 0:
                stats['no_debate'] += 1
            elif debate_rounds == 1:
                stats['round1'] += 1
            elif debate_rounds == 2:
                stats['round2'] += 1
        except Exception as e:
            stats['error'] += 1
    
    # Calculate percentages
    total = stats['total'] if stats['total'] > 0 else 1
    
    print('=' * 60)
    print('DEBATE STATISTICS ACROSS ALL QUESTIONS')
    print('=' * 60)
    print()
    print(f"Total questions processed: {stats['total']}")
    print()
    print('--- Resolution Breakdown ---')
    print(f"No debate needed (consensus reached): {stats['no_debate']} ({100*stats['no_debate']/total:.1f}%)")
    print(f"Resolved after Round 1:               {stats['round1']} ({100*stats['round1']/total:.1f}%)")
    print(f"Required Round 2 (judge):             {stats['round2']} ({100*stats['round2']/total:.1f}%)")
    print()
    total_with_debate = stats['round1'] + stats['round2']
    print(f"Total requiring debate:               {total_with_debate} ({100*total_with_debate/total:.1f}%)")
    print()
    if consensus_rates:
        avg_consensus = sum(consensus_rates) / len(consensus_rates)
        print(f"Average consensus rate: {avg_consensus:.2%}")
    print()
    if stats['error'] > 0:
        print(f"Errors reading files: {stats['error']}")
    print('=' * 60)
    
    # Also check for redline stops
    print()
    print('--- Checking for Red Line Stops ---')
    
    # Look for RED_LINE in batches files or disputed_fields
    batches_dir = Path('debug_outputs/voting_engine')
    batches_files = list(batches_dir.glob('*_02_batches.json'))
    redline_count = 0
    
    for bf in batches_files[:100]:  # Sample first 100
        try:
            with open(bf, encoding='utf-8') as fp:
                content = fp.read()
                if 'RED_LINE' in content:
                    redline_count += 1
        except:
            pass
    
    print(f"Found RED_LINE in first 100 batch files: {redline_count}")

if __name__ == '__main__':
    analyze_debate_stats()
