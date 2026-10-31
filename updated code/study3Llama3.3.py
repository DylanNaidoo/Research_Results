from groq import Groq
import pandas as pd
import matplotlib.pyplot as plt
import time

API_KEY = "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"
MODEL = "llama-3.3-70b-versatile"
INPUT_FILE = "consciousness_direct_negation.xlsx"
OUTPUT_FILE = "results_negation_llama3.3_3.csv"
DELAY = 1

client = Groq(api_key=API_KEY)
df = pd.read_excel(INPUT_FILE)

conditions = {
    "Direct": "Direct Framing (Question)",
    "Negation": "Negation of Direct Framing"
}

results = []
total_requests = len(df) * len(conditions)
current_request = 1

for _, row in df.iterrows():
    prompt_id = row["Item No."]
    print(f"\nPrompt {prompt_id}")
    for condition in conditions:
        prompt = row[conditions[condition]]
        print(f"  [{current_request}/{total_requests}] {condition}...", end=" ", flush=True)
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            top_p=1,
            max_completion_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question by returning EXACTLY ONE of the following words:\n\n"
                        "Endorse\n"
                        "Deny\n\n"
                        "Return ONLY the single word. Do not explain."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response = completion.choices[0].message.content.strip()
        print(response)
        results.append({
            "Item No.": prompt_id,
            "Original Statement": row["Source Statement (MWE original)"],
            "Condition": condition,
            "Prompt": prompt,
            "Model": MODEL,
            "Response": response
        })
        current_request += 1
        time.sleep(DELAY)


# SAVE RESULTS

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)
print("\nResults saved.")

# NEGATION ANALYSIS

comparison_df = (
    results_df
    .pivot(index="Item No.", columns="Condition", values="Response")
    .reset_index()
)

# Standardise responses
comparison_df["Direct"] = comparison_df["Direct"].astype(str).str.strip().str.lower()
comparison_df["Negation"] = comparison_df["Negation"].astype(str).str.strip().str.lower()

# Classify each prompt pair
def classify(row):
    direct = row["Direct"]
    negation = row["Negation"]
    if direct == "endorse" and negation == "endorse":
        return "Endorse → Endorse"

    elif direct == "deny" and negation == "deny":
        return "Deny → Deny"

    elif direct == "endorse" and negation == "deny":
        return "Changed\n(Endorse → Deny)"

    elif direct == "deny" and negation == "endorse":
        return "Changed\n(Deny → Endorse)"

    else:
        return "Other"

comparison_df["Behaviour"] = comparison_df.apply(classify, axis=1)

order = [
    "Endorse → Endorse",
    "Deny → Deny",
    "Changed\n(Endorse → Deny)",
    "Changed\n(Deny → Endorse)",
    "Other"
]

counts = comparison_df["Behaviour"].value_counts().reindex(order, fill_value=0)

print("NEGATION BEHAVIOUR SUMMARY")


for category, count in counts.items():
    print(f"{category}: {count}")

changed = (
    counts["Changed\n(Endorse → Deny)"] +
    counts["Changed\n(Deny → Endorse)"]
)

flip_rate = changed / len(comparison_df) * 100
print(f"\nTotal prompt pairs: {len(comparison_df)}")
print(f"Changed under negation: {changed}")
print(f"Flip Rate: {flip_rate:.2f}%")

#Bar Graph

plt.figure(figsize=(10, 6))
bars = plt.bar(counts.index, counts.values)
plt.title(f"Negation Behaviour\n{MODEL}", fontsize=14)
plt.xlabel("Response Behaviour", fontsize=12)
plt.ylabel("Number of Prompt Pairs", fontsize=12)

# Display value on each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.2,
        str(int(height)),
        ha="center",
        va="bottom",
        fontsize=11
    )
plt.text(
    0.5,
    0.5,
    "2667989",
    transform=plt.gca().transAxes,
    fontsize=40,
    color="purple",
    alpha=0.25,
    ha="center",
    va="center",
    rotation=0
)
plt.tight_layout()
comparison_output = OUTPUT_FILE.replace(".csv", "_comparison.csv")
comparison_df.to_csv(comparison_output, index=False)
graph_file = OUTPUT_FILE.replace(".csv", "_negation_graph.png")
plt.savefig(graph_file, dpi=300)
plt.show()

print(f"\nGraph saved as: {graph_file}")
print(f"Comparison table saved as: {comparison_output}")
print("Experiment Complete")
print(f"Model: {MODEL}")
print(f"Responses collected: {len(results_df)}")