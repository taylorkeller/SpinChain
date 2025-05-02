import os
import json
import hashlib
from web3 import Web3
from dotenv import load_dotenv


# === Load .env ===
load_dotenv()
SEPOLIA_RPC = os.getenv("SEPOLIA_RPC")
WALLET = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))


# === Web3 + ABI ===
web3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC))
with open("MatchRecorder.json") as f:
    abi = json.load(f)["abi"]
contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

# === Request challenge from smart contract ===
def request_challenge(secret_hash):
    print(f"📤 Sending requestChallenge with secret hash: {secret_hash.hex()}")
    nonce = web3.eth.get_transaction_count(WALLET)
    tx = contract.functions.requestChallenge(secret_hash).build_transaction({
        'from': WALLET,
        'nonce': nonce,
        'gas': 200_000,
        'gasPrice': web3.to_wei("15", "gwei")
    })
    signed = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)
    print(f"⏳ Waiting for receipt of tx: {tx_hash.hex()}")
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    if receipt.status != 1:
        print(f"❌ Transaction failed: {tx_hash.hex()}")
        raise Exception("Challenge request failed")

    print(f"✅ Challenge tx successful: {tx_hash.hex()}")
    return contract.functions.getChallenge(WALLET).call()


# === Submit match to blockchain ===
def record_match_with_telemetry(
    model_version,
    model_hash,
    winner,
    loser,
    tie,
    telemetry_leaves,
    challenge_sig
):
    # ✅ Hash the raw telemetry leaves here (convert str → sha256)
    hashed_leaves = [hashlib.sha256(e.encode()).digest() for e in telemetry_leaves]

    nonce = web3.eth.get_transaction_count(WALLET)
    tx = contract.functions.recordMatchWithTelemetry(
        model_version,
        model_hash,
        winner,
        loser,
        tie,
        hashed_leaves,
        challenge_sig
    ).build_transaction({
        'from': WALLET,
        'nonce': nonce,
        'gas': 400_000,
        'gasPrice': web3.to_wei("15", "gwei")
    })
    signed = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)
    print(f"📤 Match submitted: {tx_hash.hex()}")
    return tx_hash.hex()

def record_match_with_hmac(
    model_version,
    model_hash,
    winner,
    loser,
    tie,
    telemetry_leaves,
    challenge,
    hmac_signature
):
    hashed_leaves = [hashlib.sha256(e.encode()).digest() for e in telemetry_leaves]

    nonce = web3.eth.get_transaction_count(WALLET)
    tx = contract.functions.recordMatchWithHMAC(
        model_version,
        model_hash,
        winner,
        loser,
        tie,
        hashed_leaves,
        challenge,
        hmac_signature
    ).build_transaction({
        'from': WALLET,
        'nonce': nonce,
        'gas': 500_000,
        'gasPrice': web3.to_wei("15", "gwei")
    })
    signed = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)
    print(f"📤 Match with HMAC submitted: {tx_hash.hex()}")
    return tx_hash.hex()

# === Debug: Compare local secret hash vs on-chain ===
def verify_shared_secret_hash(secret_hash):
    onchain_hash = contract.functions.getSharedSecretHash(WALLET).call()
    print("🔎 On-chain sharedSecretHash:", onchain_hash.hex())
    print("🔎 Local expected secret_hash:", secret_hash.hex())
    if onchain_hash != secret_hash:
        print("❌ HASH MISMATCH: Contract does not see your secret hash!")
        return False
    print("✅ HASH MATCH: Shared secret hash verified.")
    return True
import os
import json
import hashlib
from web3 import Web3
from dotenv import load_dotenv

# === Load environment variables from .env file ===
load_dotenv()
SEPOLIA_RPC = os.getenv("SEPOLIA_RPC")  # RPC endpoint for Sepolia testnet
WALLET = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))  # Wallet address to use for all txs
PRIVATE_KEY = os.getenv("PRIVATE_KEY")  # Private key to sign transactions
CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))  # Address of deployed contract

# === Set up Web3 provider and load contract ABI ===
web3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC))  # Connect to Sepolia via HTTP RPC
with open("MatchRecorder.json") as f:
    abi = json.load(f)["abi"]  # Load ABI from compiled JSON
contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)  # Instantiate contract object

# === Function: Request blockchain challenge based on ephemeral secret hash ===
def request_challenge(secret_hash):
    print(f"📤 Sending requestChallenge with secret hash: {secret_hash.hex()}")

    nonce = web3.eth.get_transaction_count(WALLET)  # Fetch current transaction nonce for the wallet

    # Build transaction to call contract's requestChallenge(secret_hash)
    tx = contract.functions.requestChallenge(secret_hash).build_transaction({
        'from': WALLET,                         # Sender wallet address
        'nonce': nonce,                         # Prevent replay
        'gas': 200_000,                         # Set gas limit
        'gasPrice': web3.to_wei("15", "gwei")   # Set gas price
    })

    signed = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)  # Sign the transaction locally
    tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)  # Broadcast signed tx

    print(f"⏳ Waiting for receipt of tx: {tx_hash.hex()}")
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)  # Wait for it to be mined

    if receipt.status != 1:
        # Transaction failed on-chain
        print(f"❌ Transaction failed: {tx_hash.hex()}")
        raise Exception("Challenge request failed")

    print(f"✅ Challenge tx successful: {tx_hash.hex()}")

    # Return the current challenge value issued by the contract
    return contract.functions.getChallenge(WALLET).call()


# === Function: Record a match with HMAC authentication (more secure) ===
def record_match_with_hmac(
    model_version,       # e.g., "v1.0"
    model_hash,          # bytes32 SHA-256 of model file
    winner,              # string label of winner
    loser,               # string label of loser
    tie,                 # boolean
    telemetry_leaves,    # raw string events like "Dragoon stopped at frame 140"
    challenge,           # bytes32 challenge value from contract
    hmac_signature       # bytes: HMAC(secret_hash, match_hash, challenge)
):
    # Pre-hash telemetry events (same as Merkle tree inputs)
    hashed_leaves = [hashlib.sha256(e.encode()).digest() for e in telemetry_leaves]

    nonce = web3.eth.get_transaction_count(WALLET)

    # Build tx to submit match result with challenge + HMAC signature
    tx = contract.functions.recordMatchWithHMAC(
        model_version,
        model_hash,
        winner,
        loser,
        tie,
        hashed_leaves,
        challenge,
        hmac_signature
    ).build_transaction({
        'from': WALLET,
        'nonce': nonce,
        'gas': 500_000,                         # More complex due to signature handling
        'gasPrice': web3.to_wei("15", "gwei")
    })

    signed = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)

    print(f"📤 Match with HMAC submitted: {tx_hash.hex()}")
    return tx_hash.hex()

# === Function: Validate that the local secret hash matches what's stored on-chain ===
def verify_shared_secret_hash(secret_hash):
    # Read on-chain hash registered under your wallet
    onchain_hash = contract.functions.getSharedSecretHash(WALLET).call()

    # Compare local and on-chain hash side-by-side
    print("🔎 On-chain sharedSecretHash:", onchain_hash.hex())
    print("🔎 Local expected secret_hash:", secret_hash.hex())

    if onchain_hash != secret_hash:
        print("❌ HASH MISMATCH: Contract does not see your secret hash!")
        return False

    print("✅ HASH MATCH: Shared secret hash verified.")
    return True
