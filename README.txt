
SpinChain: Real-Time Beyblade Tracking & Blockchain Logger
===========================================================

This project tracks Beyblade matches using YOLOv5, detects the winner automatically based on spin-stop timing, and logs results to the Ethereum Sepolia blockchain with cryptographic integrity. It also provides data analytics and heatmaps of Beyblade win statistics.

-------------
 Project Structure
-------------
- spinchain.py           # Real-time match detection and blockchain submission
- blockchain_logger.py   # Smart contract interface for match recording
- Beyblades_analysis.py  # Pulls on-chain logs and generates win-rate analytics
- MatchRecorder.json     # ABI for the deployed smart contract
- .env                   # Environment variables (private key, wallet, RPC)

-------------
 Requirements 
-------------
- Python 3.8+
- YOLOv5 (Ultralytics)
- PyTorch (GPU recommended)
- OpenCV (cv2)
- Web3.py
- python-dotenv
- seaborn / matplotlib
- eth-utils, eth-abi
- pyzbar, qrcode
- webcam

-------------
How to Run
-------------

0.  Ensure your `.env` is filled out correctly.
    The contract does not need to be changed however you need to put in
    your own wallet, keys, and RPC

1. Install dependencies:
   pip install -r requirements.txt

2. Run the tracker to detect and submit matches:
   spinchain.py

   - Displays a QR challenge (rotates each match).
   - Uses YOLOv5 to detect Beyblades.
   - Tracks stability to determine who stops first.
   - Submits the result with HMAC-bound telemetry.

3. Run analytics and heatmap generation:
   Beyblades_analysis.py

   - Pulls past matches from the Sepolia blockchain.
   - Computes win/loss rates by Beyblade and class.
   - Saves heatmaps and CSVs:
     - beyblade_win_percentages.csv
     - beyblade_class_summary.csv
     - class_vs_class_heatmap.png, etc.

-------------
Testing / Debugging
-------------
- spinchain.py prints all decisions live:
  - [QR Decode], [Frame X], and match results.
- blockchain_logger.py provides:
  - verify_shared_secret_hash(...)
  - record_match_with_hmac(...)
- View blockchain logs on Etherscan using the contract address.
- Run Beyblades_analysis.py to view current win/loss rates

-------------
Security Notes
-------------
- Each match uses a fresh 256-bit ephemeral secret.
- Match results are bound via HMAC and on-chain challenge.
- Match telemetry is Merkle-hashable for audit trails.

-------------
Files Required at Runtime
-------------
- spinchain.py
- blockchain_logger.py
- MatchRecorder.json
- .env
- YOLOv5 weights at:
  yolov5/runs/train/spinchain-yolo/weights/best.pt

-------------
YOLO Training Notes
-------------
- Model expects labeled Beyblades with class names like:
  - LeonClaw, DranzerSpiral, PheonixWing, etc.
- If retraining to add Beyblades, use ultralytics/yolov5 and maintain the same label structure.

-------------
License
-------------
This system is part of a research prototype for blockchain-secured computer vision. Not intended for public deployment without security review.
