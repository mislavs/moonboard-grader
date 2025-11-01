# Data Directory

Place your training data here.

## Expected Format

Data files should be in JSON format containing an array of problem objects:

```json
[
  {
    "grade": "6B+",
    "moves": [
      {"Description": "A5", "IsStart": true, "IsEnd": false},
      {"Description": "F7", "IsStart": false, "IsEnd": false},
      {"Description": "K12", "IsStart": false, "IsEnd": true}
    ]
  }
]
```

## Files

- `problems.json` - Main training dataset (not included, generate or download separately)
- Add your own datasets here

## Data Collection

You can collect Moonboard problem data from:
- Moonboard app exports
- Web scraping (respect terms of service)
- Manual entry for specific problems

Ensure each problem has:
- A valid Font grade (5+ to 8C+)
- At least one start hold (`IsStart: true`)
- At least one end hold (`IsEnd: true`)
- Valid hold positions (columns A-K, rows 1-18)

