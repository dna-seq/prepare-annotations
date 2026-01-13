import os
import subprocess
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def undo_last_hf_commit():
    """
    Safely undoes the last commit on a Hugging Face dataset repository
    without downloading the actual data (using a blobless clone).
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN not found in .env. Please make sure it is set.")
        return

    repo_id = "just-dna-seq/ensembl_variations"
    # Using oauth2 format for the token in the URL
    clone_url = f"https://oauth2:{token}@huggingface.co/datasets/{repo_id}"
    
    tmp_dir = Path("tmp_revert_repo")
    
    # Ensure a clean state for the temporary clone
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        
    try:
        # 1. Blobless clone: Only downloads git metadata and commit history.
        # This avoids downloading gigabytes of genomic parquet files.
        print(f"🚀 Cloning {repo_id} history (metadata only)...")
        subprocess.run([
            "git", "clone", 
            "--filter=blob:none", 
            "--no-checkout", 
            "--quiet",
            clone_url, 
            str(tmp_dir)
        ], check=True)
        
        # 2. Identify the commit being removed for confirmation
        log_process = subprocess.run(
            ["git", "log", "-1", "--format=%h: %s"], 
            cwd=tmp_dir, capture_output=True, text=True, check=True
        )
        print(f"⚠️  About to remove commit: {log_process.stdout.strip()}")
        
        # 3. Force push the parent commit (HEAD~1) to the main branch.
        # This resets the remote branch to its previous state.
        print("⏪ Reverting remote to HEAD~1 (force pushing)...")
        subprocess.run(
            ["git", "push", "origin", "HEAD~1:main", "--force"], 
            cwd=tmp_dir, 
            check=True
        )
        
        print(f"✅ Success! The last commit has been removed from {repo_id}")
        print(f"🔗 Check history: https://huggingface.co/datasets/{repo_id}/commits/main")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during git operation: {e}")
        if e.stderr:
            print(f"Details: {e.stderr}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # 4. Cleanup the temporary metadata-only clone
        if tmp_dir.exists():
            print("🧹 Cleaning up temporary files...")
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    undo_last_hf_commit()




