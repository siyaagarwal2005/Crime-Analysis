name: Evaluate Submissions

on:
  push:
    paths:
      - "submission/**.csv"

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: pip install pandas scikit-learn

      - name: Write ground truth
        run: echo "${{ secrets.TEST_LABELS }}" > data/test_with_labels.csv

      - name: Run evaluator
        run: python grader/evaluator.py

      - name: Update leaderboard
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/leaderboard.json submission/results.csv
          git diff --cached --quiet || git commit -m "🤖 Update leaderboard"
          git push
