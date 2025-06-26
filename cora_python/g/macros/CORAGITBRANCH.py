import subprocess
import os
import warnings

def CORAGITBRANCH():
    """
    CORAGITBRANCH - returns the git branch of CORA

    Returns:
        str: git branch of CORA, or an empty string if not found.
    """
    
    # It's better to use a more specific custom warning class if available.
    # from cora_python.g.functions.verbose.warnings import CORAwarning
    # For now, we use a standard warning.
    def CORAwarning(identifier, message):
        warnings.warn(f"[{identifier}] {message}")

    gitbranch = ''
    curr_dir = os.getcwd()

    try:
        # Assuming CORAROOT is defined elsewhere and returns the root path
        # from cora_python.g.macros.CORAROOT import CORAROOT
        # cora_root = CORAROOT()
        # For now, let's assume the script is run from within the repo,
        # so we can search upwards for the .git directory.
        
        # A simple way to find the repo root
        path = os.getcwd()
        while path != os.path.dirname(path):
            if '.git' in os.listdir(path):
                cora_root = path
                break
            path = os.path.dirname(path)
        else:
            cora_root = None

        if cora_root is None:
            raise FileNotFoundError("'.git' directory not found.")
            
        os.chdir(cora_root)

        # Execute the git command to get the current branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True  # This will raise CalledProcessError if the command fails
        )
        gitbranch = result.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Handle cases where git is not installed, not in a repo, or other errors.
        CORAwarning('CORA:global', 'CORA does not have a git repository associated with its root folder or git is not installed.')

    finally:
        os.chdir(curr_dir)

    return gitbranch

if __name__ == '__main__':
    # Example usage:
    print(f"Current Git Branch: {CORAGITBRANCH()}") 