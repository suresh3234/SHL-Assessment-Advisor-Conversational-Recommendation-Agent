import os
import sys
import re
import json
from typing import List, Dict, Any

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastapi.testclient import TestClient
from app.main import app

# Load catalog to check for hallucinations
catalog_path = os.path.join(project_root, "app", "catalog", "catalog.json")
with open(catalog_path, "r", encoding="utf-8") as f:
    catalog_data = json.load(f, strict=False)
catalog_urls = {item["link"] for item in catalog_data}

client = TestClient(app)

def parse_trace_file(filepath: str) -> Dict[str, Any]:
    """
    Parses a conversation trace markdown file.
    Extracts:
    - User messages
    - Expected shortlist URLs from the tables
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Extract user messages
    # User messages are in blocks like:
    # **User**
    #
    # > message
    user_messages = []
    user_blocks = re.findall(r"\*\*User\*\*\s*\n+\s*>\s*(.*)", content)
    for block in user_blocks:
        user_messages.append(block.strip())
        
    # Extract expected URLs from tables
    # Match markdown table links like [link](url) or raw URLs in the table
    expected_urls = []
    # Look for shl product catalog URLs
    urls = re.findall(r"https://www.shl.com/products/product-catalog/view/[a-zA-Z0-9-]+/", content)
    # De-duplicate while preserving order
    for url in urls:
        if url not in expected_urls:
            expected_urls.append(url)
            
    return {
        "user_messages": user_messages,
        "expected_urls": expected_urls
    }

def run_evaluation():
    trace_dir = os.path.join(project_root, "GenAI_SampleConversations")
    trace_files = [f"C{i}.md" for i in range(1, 11)]
    
    results = []
    all_recalls = []
    total_hallucinations = 0
    
    print("Starting evaluation on 10 conversation traces...")
    
    for filename in trace_files:
        filepath = os.path.join(trace_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found.")
            continue
            
        trace = parse_trace_file(filepath)
        user_messages = trace["user_messages"]
        expected_urls = trace["expected_urls"]
        
        messages = []
        final_recs = []
        turns_used = 0
        transcript = []
        schema_valid = True
        
        for turn_idx, user_msg in enumerate(user_messages):
            turns_used += 1
            messages.append({"role": "user", "content": user_msg})
            transcript.append(f"User: {user_msg}")
            
            # Call chat endpoint
            try:
                response = client.post("/chat", json={"messages": messages})
                if response.status_code != 200:
                    schema_valid = False
                    break
                    
                data = response.json()
                # Verify schema contracts
                if not all(k in data for k in ("reply", "recommendations", "end_of_conversation")):
                    schema_valid = False
                    
                reply = data.get("reply", "")
                recs = data.get("recommendations", [])
                end_conv = data.get("end_of_conversation", False)
                
                transcript.append(f"Agent: {reply}")
                messages.append({"role": "assistant", "content": reply})
                
                if recs:
                    final_recs = recs
                    break
                    
                if end_conv:
                    break
            except Exception as e:
                print(f"Error during turn {turn_idx+1} of {filename}: {str(e)}")
                schema_valid = False
                break
                
        # Compute Recall@10
        retrieved_urls = [r["url"] for r in final_recs]
        # Normalize URLs (strip trailing slashes for robust comparison)
        norm_retrieved = {u.rstrip("/") for u in retrieved_urls}
        norm_expected = {u.rstrip("/") for u in expected_urls}
        
        intersection = norm_retrieved & norm_expected
        recall = len(intersection) / len(norm_expected) if norm_expected else 0.0
        all_recalls.append(recall)
        
        # Hallucination check
        hallucinated_urls = []
        for r_url in retrieved_urls:
            # Match against catalog URLs (normalized)
            norm_r_url = r_url.rstrip("/")
            if not any(norm_r_url == cat_url.rstrip("/") for cat_url in catalog_urls):
                hallucinated_urls.append(r_url)
                
        total_hallucinations += len(hallucinated_urls)
        
        results.append({
            "trace": filename,
            "recall": recall,
            "turns_used": turns_used,
            "retrieved_count": len(retrieved_urls),
            "expected_count": len(expected_urls),
            "hallucinations": hallucinated_urls,
            "schema_valid": schema_valid,
            "transcript": transcript
        })
        
        print(f"Trace {filename}: Recall={recall:.2f}, Turns={turns_used}, Hallucinations={len(hallucinated_urls)}")
        
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    print(f"\nEvaluation Complete. Mean Recall@10: {mean_recall:.4f}, Total Hallucinations: {total_hallucinations}")
    
    # Generate Markdown Report
    generate_report(results, mean_recall, total_hallucinations)

def generate_report(results: List[Dict[str, Any]], mean_recall: float, total_hallucinations: int):
    report_path = os.path.join(project_root, "scripts", "eval_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SHL Assessment Recommendation Agent - Evaluation Report\n\n")
        
        # Summary
        f.write("## Summary Metrics\n\n")
        f.write(f"- **Mean Recall@10**: {mean_recall:.4f}\n")
        f.write(f"- **Total Hallucinated URLs**: {total_hallucinations}\n")
        f.write(f"- **Schema Validity**: {'100% Valid' if all(r['schema_valid'] for r in results) else 'Failed'}\n\n")
        
        # Summary Table
        f.write("### Per-Trace Results\n\n")
        f.write("| Trace | Recall@10 | Turns Used | Retrieved | Expected | Hallucinations | Schema Valid |\n")
        f.write("|-------|-----------|------------|-----------|----------|----------------|--------------|\n")
        for r in results:
            hall_str = "None" if not r["hallucinations"] else f"{len(r['hallucinations'])} ({', '.join(r['hallucinations'])})"
            f.write(f"| {r['trace']} | {r['recall']:.2f} | {r['turns_used']} | {r['retrieved_count']} | {r['expected_count']} | {hall_str} | {r['schema_valid']} |\n")
            
        f.write("\n## Detailed Transcripts\n\n")
        for r in results:
            f.write(f"### {r['trace']}\n\n")
            f.write("```\n")
            for line in r["transcript"]:
                f.write(f"{line}\n")
            f.write("```\n\n")
            
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_evaluation()
