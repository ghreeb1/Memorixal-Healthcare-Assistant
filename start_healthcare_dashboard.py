#!/usr/bin/env python3
"""
Healthcare Dashboard Startup Script
This script starts the unified healthcare monitoring dashboard.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'sqlalchemy': 'sqlalchemy',
        'pydantic': 'pydantic',
    }
    
    missing_packages = []
    
    print("Checking dependencies...")
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"  ✗ {package_name}")
    
    if missing_packages:
        print("\nMissing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_project_structure():
    """Check if all required files exist."""
    required_files = ['main.py']
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Required files not found: {', '.join(missing_files)}")
        print("Please run this script from the project root directory")
        return False
    
    return True

def display_banner():
    """Display startup banner with server information."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║         Healthcare Monitoring Dashboard                      ║
║                                                              ║
║  Patient & Caregiver Activity Tracking System               ║
╚══════════════════════════════════════════════════════════════╝

Server will be available at:
  → Patient Interface:    http://localhost:8000/patient
  → Caregiver Dashboard:  http://localhost:8000/dashboard
  → Role Selection:       http://localhost:8000/role-selection
  → API Documentation:    http://localhost:8000/docs

Database: SQLite (dashboard_data.db)
Auto-reload: Enabled

Press Ctrl+C to stop the server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(banner)

def start_server():
    """Start the FastAPI server."""
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0",  # Allow external connections
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        print("Thank you for using Healthcare Monitoring Dashboard!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Check if port 8000 is already in use")
        print("2. Ensure all dependencies are installed")
        print("3. Check main.py for syntax errors")
        return False
    except FileNotFoundError:
        print("\nError: uvicorn not found")
        print("Install it with: pip install uvicorn")
        return False
    
    return True

def main():
    """Main startup function."""
    # Check project structure
    if not check_project_structure():
        sys.exit(1)
    
    # Check dependencies
    print("\n" + "="*60)
    if not check_dependencies():
        sys.exit(1)
    
    print("\nAll dependencies found!")
    print("="*60)
    
    # Display banner
    display_banner()
    
    # Start server
    success = start_server()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()