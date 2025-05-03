import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from web3 import Web3
from dotenv import load_dotenv
import unicodedata

# Load environment variables
load_dotenv()
INFURA_URL = os.getenv("SEPOLIA_RPC")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")

# Load contract ABI
with open("MatchRecorder.json", "r") as f:
    abi = json.load(f)

# Connect to Sepolia
web3 = Web3(Web3.HTTPProvider(INFURA_URL))
contract = web3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi["abi"])

event_signature = "MatchWithHMAC(address,string,bytes32,string,string,bool,bytes32,bytes32[],bytes,uint256)"
event_topic =  web3.keccak(text=event_signature).hex()


# Get logs
logs = web3.eth.get_logs({
    "fromBlock": 0,
    "toBlock": "latest",
    "address": Web3.to_checksum_address(CONTRACT_ADDRESS),
    "topics": [event_topic]
})
events = [contract.events.MatchWithHMAC().process_log(log) for log in logs]
print(f"✅ Logs fetched: {len(events)}")

# Clean + validate names (removes nulls, control characters, and invalid names)
def safe_str(val):
    if isinstance(val, bytes):
        val = val.decode('utf-8', errors='ignore').rstrip('\x00')
    if not isinstance(val, str):
        return None
    val = val.strip()
    val = ''.join(c for c in val if unicodedata.category(c)[0] != "C")  # remove control chars
    if len(val) < 2:
        return None
    safe_chars = sum(c.isalnum() or c in " _-" for c in val)
    if safe_chars / len(val) < 0.6:
        return None
    return val

# Parse logs into DataFrame
data = []
for evt in events:
    args = evt["args"]
    w = safe_str(args["winner"])
    l = safe_str(args["loser"])
    t = "Yes" if args["tie"] else "No"

    b1 = w
    b2 = l

    # Only include matches with valid winner and loser names
    if None not in [b1, b2, w, l, t]:
        data.append({
            "beyblade1": b1,
            "beyblade2": b2,
            "winner": w,
            "loser": l,
            "tie": t,
            "timestamp": args["timestamp"]
        })

df = pd.DataFrame(data)
# Stop here if matches were pulled
if df.empty:
    print("⚠️ No valid match data found after filtering.")
else:
    # Predefined class for named Beyblades
    beyblade_classes = {
        "LeonClaw": "Defense",
        "PheonixWing": "Attack",
        "DranzerSpiral": "Stamina",
        "ShinobiShadow": "Defense",
        "DranDagger": "Attack",
    }

    # Get a list of unique Beyblade names
    beys = pd.unique(pd.concat([df['beyblade1'], df['beyblade2'], df['winner'], df['loser']]))

    # Initialize win/loss/tie summary
    summary = pd.DataFrame(0, index=beys, columns=["Wins", "Losses", "Ties"])
    for _, row in df.iterrows():
        if row["tie"] == "Yes":
            summary.loc[row["beyblade1"], "Ties"] += 1
            summary.loc[row["beyblade2"], "Ties"] += 1
        else:
            summary.loc[row["winner"], "Wins"] += 1
            summary.loc[row["loser"], "Losses"] += 1

    # Add total matches and win percentage
    summary["Total Matches"] = summary.sum(axis=1)
    summary["Win %"] = (summary["Wins"] / summary["Total Matches"]).round(2)
    summary = summary[summary["Total Matches"] > 0]
    summary["Class"] = summary.index.map(lambda x: beyblade_classes.get(x, "Unknown"))

    # Output summary by individual Beyblade
    print("\n📊 Beyblade Win Percentages:\n")
    print(summary.sort_values("Win %", ascending=False))
    summary.to_csv("beyblade_win_percentages.csv")

    # Summarize by class 
    class_summary = summary.groupby("Class")[["Wins", "Losses", "Ties", "Total Matches"]].sum()
    class_summary["Win %"] = (class_summary["Wins"] / class_summary["Total Matches"]).round(2)
    print("\n📊 Win Percentages by Class:\n")
    print(class_summary.sort_values("Win %", ascending=False))
    class_summary.to_csv("beyblade_class_summary.csv")

    # Map winner and loser classes
    df["class_winner"] = df["winner"].map(lambda x: beyblade_classes.get(x, "Unknown"))
    df["class_loser"] = df["loser"].map(lambda x: beyblade_classes.get(x, "Unknown"))
    # Remove matches with Unknown class (None-category)
    df = df[(df["class_winner"] != "Unknown") & (df["class_loser"] != "Unknown")]

    # Build a matrix of class-vs-class wins
    classes = sorted(set(df["class_winner"]).union(set(df["class_loser"])))
    class_matrix = pd.DataFrame(0, index=classes, columns=classes)

    for _, row in df.iterrows():
        if row["tie"] == "No":
            class_matrix.loc[row["class_winner"], row["class_loser"]] += 1

    class_matrix.to_csv("class_vs_class_matrix.csv")
    class_winrate = class_matrix.div(class_matrix.sum(axis=1), axis=0).fillna(0).round(2)
    class_winrate.to_csv("class_vs_class_winrate.csv")

    # Heatmap of class vs class match counts
    plt.figure(figsize=(8, 6))
    sns.heatmap(class_matrix, annot=True, fmt="d", cmap="YlOrRd", cbar=True)
    plt.title("Class vs. Class Win Matrix")
    plt.xlabel("Loser Class")
    plt.ylabel("Winner Class")
    plt.tight_layout()
    plt.savefig("class_vs_class_heatmap.png")
    plt.show()

    # Heatmap of class win rates
    plt.figure(figsize=(8, 6))
    sns.heatmap(class_winrate, annot=True, fmt=".2f", cmap="BuGn", cbar=True)
    plt.title("Class vs. Class Win Rates")
    plt.xlabel("Loser Class")
    plt.ylabel("Winner Class")
    plt.tight_layout()
    plt.savefig("class_vs_class_winrate_heatmap.png")
    plt.show()

    # Beyblade vs Beyblade Matrix
    unique_beys = sorted(set(df['winner']).union(df['loser']))
    name_matrix = pd.DataFrame(0, index=unique_beys, columns=unique_beys)

    for _, row in df.iterrows():
        if row["tie"] == "No":
            name_matrix.loc[row["winner"], row["loser"]] += 1

    name_matrix.to_csv("beyblade_vs_beyblade_matrix.csv")

    name_winrate = name_matrix.div(name_matrix.sum(axis=1), axis=0).fillna(0).round(2)
    name_winrate.to_csv("beyblade_vs_beyblade_winrate.csv")

    # Plotting Beyblade vs Beyblade win matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(name_matrix, annot=True, fmt="d", cmap="PuBu", cbar=True)
    plt.title("Beyblade vs. Beyblade Win Matrix")
    plt.xlabel("Loser")
    plt.ylabel("Winner")
    plt.tight_layout()
    plt.savefig("beyblade_vs_beyblade_heatmap.png")
    plt.show()


    # Plotting Beyblade vs Beyblade win rate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(name_winrate, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("Beyblade vs. Beyblade Win Rate Matrix")
    plt.xlabel("Loser")
    plt.ylabel("Winner")
    plt.tight_layout()
    plt.savefig("beyblade_vs_beyblade_winrate_heatmap.png")
    plt.show()


    print("\n✅ All summaries and heatmaps saved.")