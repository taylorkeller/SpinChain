import os
import json
from web3 import Web3
from dotenv import load_dotenv

# Load .env environment variables
load_dotenv()
print("Loaded RPC:", os.getenv("SEPOLIA_RPC"))

# Load from .env
INFURA_URL = os.getenv("SEPOLIA_RPC")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")


# Load the ABI
with open("MatchRecorderABI.json", "r") as f:
    abi = json.load(f)

# Connect to Sepolia via Infura
web3 = Web3(Web3.HTTPProvider(INFURA_URL))
assert web3.is_connected(), "❌ Web3 failed to connect to Sepolia"

# Prepare contract
contract = web3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=abi
)

# Record match result on-chain
def record_match(bey1, bey2, winner, loser, tie):
    nonce = web3.eth.get_transaction_count(WALLET_ADDRESS)

    txn = contract.functions.recordMatch(
        bey1, bey2, winner, loser, tie
    ).build_transaction({
        'from': WALLET_ADDRESS,
        'nonce': nonce,
        'gas': 200000,
        'gasPrice': web3.to_wei('20', 'gwei')
    })

    signed = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
    print("✅ Submitted match to blockchain:", tx_hash.hex())
    return tx_hash.hex()
