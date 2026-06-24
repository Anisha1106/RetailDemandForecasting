# Power BI Dashboard Guide

Use `powerbi/retail_demand_powerbi_dataset.csv` as the source file in Power BI Desktop.

## Recommended Report Pages

| Page | Visuals | Slicers |
| --- | --- | --- |
| Executive Overview | KPI cards, actual vs predicted sales trend, top and bottom stores | Date, store, holiday label |
| Store Drilldown | Store-level trend, weekly table, sales gap card | Store, date |
| Forecast Accuracy | MAE by store, forecast gap %, monthly error trend | Store, year, holiday label |
| Economic Drivers | Temperature, fuel price, CPI, unemployment compared with weekly sales | Store, year, month |

## Suggested DAX Measures

```DAX
Total Actual Sales = SUM(retail_demand_powerbi_dataset[Weekly_Sales])
Total Predicted Sales = SUM(retail_demand_powerbi_dataset[Predicted_Sales])
Forecast Gap = [Total Predicted Sales] - [Total Actual Sales]
Forecast Gap % = DIVIDE([Forecast Gap], [Total Actual Sales])
Mean Absolute Error = AVERAGE(retail_demand_powerbi_dataset[Absolute_Error])
Mean Absolute Percentage Error = AVERAGE(retail_demand_powerbi_dataset[Absolute_Percentage_Error])
Holiday Sales = CALCULATE([Total Actual Sales], retail_demand_powerbi_dataset[Holiday_Label] = "Holiday")
```

## Refresh Workflow

Run this after regenerating `data/processed/predictions.csv`:

```bash
venv/bin/python src/export_powerbi_dataset.py
```

Then refresh the CSV source inside Power BI Desktop.
