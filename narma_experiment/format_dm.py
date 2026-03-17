import pandas as pd
for crit, name in [("MSE", "S&P 500"), ("QLIKE", "S&P 500"), ("MSE", "NARMA10"), ("QLIKE", "NARMA10")]:
    file_prefix = "SP500_Realized_Volatility" if "S" in name else "NARMA10"
    df = pd.read_csv(f"/Users/umut/Desktop/EPFL_ANTI/narma_experiment/results/{file_prefix}_DM_{crit}.csv", index_col=0)
    
    print(f"\n### Diebold-Mariano Test ({name} - {crit})\n")
    cols = ["Model"] + list(df.columns)
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    
    for idx, row in df.iterrows():
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print(f"| **{idx}** | " + " | ".join([f"`{v}`" if v else "" for v in vals]) + " |")
