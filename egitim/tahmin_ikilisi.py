geohashler, pencereler = [], []
simdi = df["datetime"].max()

for gh, grp in df.groupby("geohash"):
    son6 = grp.sort_values("datetime").tail(SEQUENCE_LEN)
    # Son 6 saat eksiksiz ve güncel mi?
    if (len(son6) == SEQUENCE_LEN
            and son6["datetime"].iloc[-1] == simdi
            and son6["datetime"].iloc[-1] - son6["datetime"].iloc[0] == pd.Timedelta(hours=SEQUENCE_LEN - 1)):
        geohashler.append(gh)
        pencereler.append(scaler_X.transform(son6[FEATURE_COLS].values))

X = np.stack(pencereler)                                   # (52, 6, 23)
tahmin = scaler_y.inverse_transform(model.predict(X, verbose=0)).ravel()

sonuc = pd.DataFrame({
    "geohash": geohashler,
    "congestion_ratio": tahmin,
    "tahmin_saati": simdi + pd.Timedelta(hours=1),
})
