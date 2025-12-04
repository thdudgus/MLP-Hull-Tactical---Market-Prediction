import pandas as pd 

def build_solution(train_path: str, output_path: str = "solution.csv") -> None:
    train = pd.read_csv(train_path)
    solution = train[["date_id", "risk_free_rate", "forward_returns"]].copy()

    solution.to_csv(output_path, index=False)
    print(f"Saved solution to {output_path}")

if __name__ == "__main__":
    build_solution("train.csv")