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
