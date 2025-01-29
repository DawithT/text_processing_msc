"""
Projekt beállítási parancsfájl
Létrehozza a könyvtárstruktúrát és a kezdeti fájlokat a biomarker projekthez
"""
import os
from pathlib import Path
import subprocess
import sys

def create_directory_structure():
    """A projekt könyvtárstruktúra létrehozása"""
    # Az alapkönyvtár meghatározása aktuális munkakönyvtárként
    base_dir = Path.cwd()
    
    # A projekt fő könyvtárának létrehozása
    project_dir = base_dir / "biomarker_project"
    project_dir.mkdir(exist_ok=True)
    
    # Alkönyvtárak létrehozása
    directories = [
        project_dir / "src",
        project_dir / "output",
        project_dir / "output" / "visualizations",
        project_dir / "tests"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        
    # Látrehoz __init__.py fájlokat
    (project_dir / "src" / "__init__.py").touch()
    (project_dir / "tests" / "__init__.py").touch()
    
    # Létrehoz .env fájlokat sablon tartalommal
    env_content = """# Biomarker Analysis Configuration
ENTREZ_EMAIL=your.email@example.com
MAX_RESULTS=1000
BATCH_SIZE=100
"""
    
    with open(project_dir / ".env", "w") as f:
        f.write(env_content)
    
    # Létrehoz egy requirements.txt fájlt a szükséges csomagokkal
    requirements_content = """spacy==3.7.2
scispacy==0.5.3
pandas==2.1.4
numpy==1.24.3
scipy==1.11.4
requests==2.31.0
tenacity==8.2.3
matplotlib==3.8.2
seaborn==0.13.1
biopython==1.83
nltk==3.8.1
scikit-learn==1.3.2
openpyxl==3.1.2
python-dotenv==1.0.0
"""
    
    with open(project_dir / "requirements.txt", "w") as f:
        f.write(requirements_content)
    
    # Tesztelési szkript létrehozása
    test_script_content = """import spacy
import pandas as pd
import numpy as np
from Bio import Entrez
from pathlib import Path

def test_imports():
    # Test basic imports
    print("Basic imports successful!")
    
    # Test spaCy model loading
    try:
        nlp = spacy.load("en_core_sci_lg")
        print("SpaCy model loaded successfully!")
    except Exception as e:
        print(f"Error loading SpaCy model: {e}")
    
    # Test directory structure
    base_dir = Path(__file__).parent.parent
    required_dirs = ["src", "output", "output/visualizations"]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"Directory {dir_name} exists!")
        else:
            print(f"Warning: Directory {dir_name} not found!")

if __name__ == "__main__":
    test_imports()
"""
    
    with open(project_dir / "tests" / "test_setup.py", "w") as f:
        f.write(test_script_content)
    
    return project_dir

def setup_virtual_environment(project_dir):
    """Virtuális környezet létrehozása és konfigurálása"""
    try:
        # Virtuális környezet létrehozása
        subprocess.run([sys.executable, "-m", "venv", str(project_dir / "venv")], check=True)
        
        # A pip elérési útvonalának meghatározása az operációs rendszer alapján
        if os.name == 'nt':  # Windows
            pip_path = project_dir / "venv" / "Scripts" / "pip"
            activate_script = project_dir / "venv" / "Scripts" / "activate"
        else:  # Unix/Linux/MacOS
            pip_path = project_dir / "venv" / "bin" / "pip"
            activate_script = project_dir / "venv" / "bin" / "activate"
        
        # Követelmények telepítése
        subprocess.run([str(pip_path), "install", "-r", str(project_dir / "requirements.txt")], check=True)
        
        # A scispacy modell telepítése
        subprocess.run([str(pip_path), "install", "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz"], check=True)
        
        return activate_script
        
    except subprocess.CalledProcessError as e:
        print(f"Error setting up virtual environment: {e}")
        return None

def main():
    """Main setup function"""
    print("Starting project setup...")
    
    # Könyvtárstruktúra létrehozása
    project_dir = create_directory_structure()
    print(f"\nCreated project directory structure at: {project_dir}")
    print("\nDirectory structure created with:")
    print("- src/")
    print("- output/")
    print("  - visualizations/")
    print("- tests/")
    print("- .env")
    print("- requirements.txt")
    
    # Virtuális környezet beállítása
    print("\nSetting up virtual environment...")
    activate_script = setup_virtual_environment(project_dir)
    
    if activate_script:
        print("\nSetup completed successfully!")
        print(f"\nTo activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f"    {activate_script}")
        else:
            print(f"    source {activate_script}")
        print("\nAfter activation, run:")
        print("    python tests/test_setup.py")
    else:
        print("\nSetup completed with errors. Please check the error messages above.")

if __name__ == "__main__":
    main()