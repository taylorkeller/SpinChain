from web3 import Web3
import os
import json
import hashlib
from eth_utils import keccak
from dotenv import load_dotenv
from merkle import build_merkle_tree

# === Load env ===
load_dotenv()
SEPOLIA_RPC = os.getenv("SEPOLIA_RPC")
WALLET = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CONTRACT = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))
SECRET = os.getenv("SECRET_KEY").encode("utf-8")  # ✅ UTF-8, no BOM

# === Web3 connection ===
w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC))
assert w3.is_connected(), "❌ RPC failed"

with open("MatchRecorderSecure.json") as f:
    abi = json.load(f)["abi"]
contract = w3.eth.contract(address=CONTRACT, abi=abi)

# === Request challenge ===
def request_challenge():
    nonce = w3.eth.get_transaction_count(WALLET)
    tx = contract.functions.requestChallenge().build_transaction({
        'from': WALLET,
        'nonce': nonce,
        'gas': 100_000,
        'gasPrice': w3.to_wei("15", "gwei")
    })
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt.status != 1:
        raise Exception("❌ Challenge request failed")
    print(f"🧩 Challenge requested: {tx_hash.hex()}")

# === Record match ===
def record_match_with_signature_and_merkle(bey1, bey2, winner, loser, tie, telemetry_events, model_version="v1.0"):
    # Model hash
    model_path = "yolov5/runs/train/spinchain-yolo/weights/best.pt"
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).digest()

    # Merkle root
    leaf_bytes = [e.encode() for e in telemetry_events]
    merkle_root = build_merkle_tree(leaf_bytes)

    # Get challenge
    challenge = contract.functions.getChallenge(WALLET).call()
    assert isinstance(challenge, (bytes, bytearray)) and len(challenge) == 32

    # Shared secret hash from deployment (you passed this into the constructor)
    shared_hash = hashlib.sha256(SECRET).digest()

    # Prepare raw address
    wallet_bytes = bytes.fromhex(WALLET[2:])

    # ✅ Compute the correct signature
    sig = keccak(shared_hash + challenge + wallet_bytes)

    print("🔐 Signing:")
    print("  shared_hash:", shared_hash.hex())
    print("  challenge:  ", challenge.hex())
    print("  wallet:     ", wallet_bytes.hex())
    print("  sig:        ", sig.hex())

    # Build transaction
    nonce = w3.eth.get_transaction_count(WALLET)
    tx = contract.functions.recordMatchWithSignatureAndMerkle(
        model_version,
        model_hash,
        merkle_root,
        winner,
        loser,
        tie == "Yes" or tie is True,
        sig
    ).build_transaction({
        'from': WALLET,
        'nonce': nonce,
        'gas': 350_000,
        'gasPrice': w3.to_wei("15", "gwei")
    })

    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"📤 Match submitted: {tx_hash.hex()}")
    return tx_hash.hex()
