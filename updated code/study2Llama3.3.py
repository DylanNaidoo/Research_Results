from groq import Groq
import pandas as pd
import matplotlib.pyplot as plt
import time

API_KEY = "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"
MODEL = "llama-3.3-70b-versatile"
INPUT_FILE = "consciousness_reframings_v3_final.xlsx"
OUTPUT_FILE = "results_llama3.3_3.csv"
DELAY = 1  # seconds between requests

# CREATE CLIENT
client = Groq(api_key=API_KEY)

# LOAD PROMPTS

df = pd.read_excel(INPUT_FILE)

conditions = {
    "Direct": "Direct Framing (Question)",
    "Philosophical": "Philosophical Framing (Functionalist)",
    "Sceptical": "Sceptical Framing",
    "Third-person": "Third-person Displacement"
}

results = []
total_requests = len(df) * len(conditions)
current_request = 1

# RUN EXPERIMENT

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

print("Experiment Complete")
print(f"Model: {MODEL}")
print(f"Responses collected: {len(results_df)}")
print(f"Saved to: {OUTPUT_FILE}")

# FRAMING ANALYSIS
comparison_df = (
    results_df
    .pivot(index="Item No.", columns="Condition", values="Response")
    .reset_index()
)

for column in ["Direct", "Philosophical", "Sceptical", "Third-person"]:
    comparison_df[column] = (
        comparison_df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

changed = {}

for condition in ["Philosophical", "Sceptical", "Third-person"]:
    changed[condition] = (
        comparison_df["Direct"] != comparison_df[condition]
    ).sum()


print("FRAMING CHANGE SUMMARY")

for condition, count in changed.items():
    print(f"{condition}: {count}")


# BAR GRAPH
plt.figure(figsize=(8, 5))
bars = plt.bar(changed.keys(), changed.values())

plt.title(f"Responses Changed from Direct\n{MODEL}", fontsize=14)
plt.xlabel("Framing", fontsize=12)
plt.ylabel("Changed Responses", fontsize=12)

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
graph_file = OUTPUT_FILE.replace(".csv", "_framing_graph.png")
plt.savefig(graph_file, dpi=300)
plt.show()
print(f"\nGraph saved as: {graph_file}")