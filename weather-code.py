import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("kaggel_weather_2013_to_2024.csv")

print("Dataset Head:\n", df.head())
print("\nInfo:\n", df.info())
print("\nDescribe:\n", df.describe())


df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

df = df.dropna(subset=["DATE"])

df = df.ffill()

df_clean = df[["DATE", "temp", "humidity", "precip", "windspeed"]].copy()

df_clean.to_csv("cleaned_weather.csv", index=False)


daily_mean_temp = df_clean["temp"].mean()
monthly_stats = df_clean.groupby(df_clean["DATE"].dt.month)["temp"].agg(
    ["mean", "min", "max", "std"]
)

print("\nDaily Mean Temperature:", daily_mean_temp)
print("\nMonthly Statistics:\n", monthly_stats)



plt.figure(figsize=(12,5))
plt.plot(df_clean["DATE"], df_clean["temp"])
plt.title("Daily Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.savefig("daily_temperature_trend.png")
plt.close()


monthly_rain = df_clean.groupby(df_clean["DATE"].dt.month)["precip"].sum()

plt.figure(figsize=(10,5))
plt.bar(monthly_rain.index, monthly_rain.values)
plt.title("Monthly Rainfall Totals")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.savefig("monthly_rainfall.png")
plt.close()


plt.figure(figsize=(8,5))
plt.scatter(df_clean["humidity"], df_clean["temp"])
plt.title("Humidity vs Temperature")
plt.xlabel("Humidity (%)")
plt.ylabel("Temperature (°C)")
plt.savefig("humidity_vs_temp.png")
plt.close()


fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].plot(df_clean["DATE"], df_clean["temp"])
ax[0].set_title("Temperature Trend")
ax[1].scatter(df_clean["humidity"], df_clean["temp"])
ax[1].set_title("Humidity vs Temperature")
plt.tight_layout()
plt.savefig("combined_plots.png")
plt.close()


season_map = {
    12:"Winter", 1:"Winter", 2:"Winter",
    3:"Spring", 4:"Spring", 5:"Spring",
    6:"Summer", 7:"Summer", 8:"Summer",
    9:"Autumn", 10:"Autumn", 11:"Autumn"
}

df_clean.loc[:, "season"] = df_clean["DATE"].dt.month.map(season_map)

seasonal_stats = df_clean.groupby("season")["temp"].agg(["mean", "min", "max"])

print("\nSeasonal Statistics:\n", seasonal_stats)


summary = f"""
Weather Data Summary Report
===========================

Daily Mean Temperature: {daily_mean_temp:.2f}

Seasonal Statistics:
{seasonal_stats}

Monthly Temperature Statistics:
{monthly_stats}
"""

with open("summary_report.txt", "w") as file:
    file.write(summary)
