import subprocess

library_path = r"C:\Users\eugene.dann\Documents\dev\KiCad_Libs\LCSC"

while True:
    lcsc_id = input("Enter LCSC ID (or 'quit' to exit): ").strip()
    
    if lcsc_id.lower() in ("quit", "exit", "q"):
        print("Goodbye.")
        break
    
    if not lcsc_id:
        print("Please enter a valid LCSC ID.")
        continue

    # Build and run the command
    cmd = ["easyeda2kicad", "--full", f"--lcsc_id={lcsc_id}", "--output", library_path]
    subprocess.run(cmd)
