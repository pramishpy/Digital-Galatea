import os
import sys
import subprocess

def check_python_imports():
    """Check if required Python packages are installed"""
    required_packages = ['flask', 'dotenv', 'nltk', 'transformers', 'google-generativeai']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                __import__('python_dotenv')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_file_structure():
    """Check if all required files and directories exist"""
    expected_files = [
        'app.py',
        'import_random.py',
        'templates/index.html',
        'static/css/style.css',
        'static/js/script.js',
        '.env'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def check_env_file():
    """Check if .env file has GEMINI_API_KEY"""
    if not os.path.exists('.env'):
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
        
    return 'GEMINI_API_KEY' in content

def main():
    print("=== Galatea Website Troubleshooter ===\n")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")
    
    # Check required packages
    print("\nChecking required packages...")
    missing_packages = check_python_imports()
    if missing_packages:
        print("❌ The following packages need to be installed:")
        for package in missing_packages:
            install_name = 'python-dotenv' if package == 'dotenv' else package
            print(f"  - {install_name}")
        print("\nInstall them using: pip install package-name")
    else:
        print("✅ All required packages are installed.")
    
    # Check file structure
    print("\nChecking file structure...")
    missing_files = check_file_structure()
    if missing_files:
        print("❌ The following files/directories are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    else:
        print("✅ All required files and directories exist.")
    
    # Check .env file
    print("\nChecking environment variables...")
    if check_env_file():
        print("✅ GEMINI_API_KEY found in .env file.")
    else:
        print("❌ GEMINI_API_KEY not found in .env file.")
        print("   Create a .env file with: GEMINI_API_KEY=your_api_key_here")
    
    print("\n=== Conclusion ===")
    if not missing_packages and not missing_files and check_env_file():
        print("✅ Everything looks good! The website should work correctly.")
        print("   Run 'python app.py' to start the server.")
        print("   Then open http://127.0.0.1:5000 in your browser.")
    else:
        print("❌ Some issues were found that need to be addressed before the website will work.")
    
    print("\nWould you like to try fixing these issues automatically? (y/n)")
    choice = input("> ")
    
    if choice.lower() == 'y':
        # Install missing packages
        if missing_packages:
            print("\nInstalling missing packages...")
            for package in missing_packages:
                install_name = 'python-dotenv' if package == 'dotenv' else package
                print(f"Installing {install_name}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', install_name])
        
        # Create missing directories
        missing_dirs = set()
        for file_path in missing_files:
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                missing_dirs.add(dir_path)
        
        for dir_path in missing_dirs:
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        
        # Create .env file if missing
        if not check_env_file():
            print("\nCreating .env file...")
            api_key = input("Enter your Gemini API Key: ")
            with open('.env', 'w') as f:
                f.write(f"GEMINI_API_KEY={api_key}\n")
        
        print("\nFixes applied. Run 'python app.py' to start the server.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
