# Swing-Trade Marktscreener

Streamlit-App zum Screenen von Sektoren auf Momentum — erst Grobdurchlauf
über die Kern-Ticker, dann Drill-Down mit erweiterter Ticker-Liste
(Small Caps) und Detail-Kennzahlen.

## Starten

```
cd swing-screener
python -m streamlit run app.py
```

Die App öffnet sich im Browser (Standard: http://localhost:8501).

## Aufbau

- **Sektor-Übersicht**: Momentum-Score pro Sektor (Durchschnitt der
  Kern-Ticker), sortiert. Zeigt auf einen Blick, wo Bewegung ist.
- **Drill-Down** (Klick auf Sektor): alle Ticker inkl. der erweiterten
  Liste, vollständige Kennzahlen-Tabelle, Candlestick-Chart mit
  SMA 10/30, Volumen und MACD.
- **Sektoren erweitern**: direkt in der App (Expander „Sektor bearbeiten“
  bzw. „Neuen Sektor anlegen“ in der Sidebar) oder per Hand in
  `sectors.json`. Europäische Ticker im Yahoo-Format angeben
  (z.B. `LPK.DE`, `SIVE.ST`).

## Kennzahlen

| Kennzahl | Bedeutung |
|---|---|
| Ø Open/Close 10T & 30T | Durchschnittspreise — Kern deiner Logik |
| Trend 10/30 | 10T-Close-Schnitt vs. 30T-Close-Schnitt (Momentum-Hauptsignal) |
| Ø Intraday | mittlere Bewegung Open→Close pro Tag |
| Ø Overnight | mittlere Bewegung Vortages-Close→Open (Gap-Verhalten) |
| Ø Tag | mittlere Close-zu-Close-Veränderung |
| SMA20/50, MACD | klassische Trendbestätigung |
| RSI | Überhitzung (>70) / überverkauft (<30) |
| Rel. Volumen | 10T-Volumen vs. 60T-Volumen — Momentum braucht Volumen |
| Abstand 52W-Hoch | Nähe zum Hoch = relative Stärke |

## Momentum-Score (−100 … +100)

Gewichtete Summe: 30 % Trend 10/30, je 12,5 % Intraday- und
Overnight-Drift, je 10 % MACD, MA-Anordnung und relatives Volumen,
15 % Nähe zum 52-Wochen-Hoch. Gewichte stehen in `compute_metrics()`
in `app.py` und lassen sich dort leicht anpassen.

Ampel: 🟢 ≥ 30 · 🟡 ≥ 10 · ⚪ ≥ −10 · 🔴 darunter — farblich hinterlegt in
Sidebar und Übersichtstabelle.

**Warum 10/30 Tage und nicht kürzer?** Im Backtest (2 Jahre, Trailing-Exit)
wurden 5/20-, 7/21- und 10/30-Fenster verglichen: 10/30 lieferte das beste
Ergebnis (Ø 1,10 R, PF 2,83) — kürzere Fenster erzeugen ~20 % mehr Trades,
aber mit schlechterer Qualität (Ø 0,87–0,94 R). Der Einstiegs-Trigger ist
ohnehin tagesaktuell (SMA10-Berührung, 20-Tage-Hoch); die 10/30-Fenster
messen nur den Trend-Hintergrund, und der ist stabiler = besser.

## Kauf-/Exit-Signale (Setup-Scanner)

Signale gelten nur in Trendrichtung (Score ≥ 10 und 10T-Schnitt > 30T-Schnitt):

| Signal | Regel |
|---|---|
| 🟢 Kauf: Pullback | SMA10 in den letzten 3 Tagen berührt, heute darüber und über dem Open geschlossen, RSI < 70 |
| 🟢 Kauf: Breakout | Schluss über dem 20-Tage-Hoch bei rel. Volumen > 1,2× |
| 🟡 Beobachten | Trend intakt, aber kein Trigger — auf Rücksetzer warten |
| 🔴 Exit | Trend war aufwärts, Schluss unter SMA20 |
| ⚫ meiden | Score ≤ −10 (Abwärtstrend) |

Zu jedem Kaufsignal: **Stop** (letztes 5-Tage-Tief bzw. 2×ATR unter Kurs, der
engere der beiden), **Ziel (2R)** bei Chance/Risiko 2:1 und **Risiko %**
(Abstand Kurs→Stop). Positionsgröße darüber steuern: riskiere pro Trade z.B.
max. 1 % des Depots, also `Stückzahl = 1 % Depot / (Kurs − Stop)`.

Der 🎯 Setup-Scanner (Sidebar) prüft alle Ticker aller Sektoren auf einmal
und zeigt nur aktive Signale. Regeln stehen in `compute_setup()` in `app.py`.

Wichtige Verfeinerungen (v2):

- **Extended-Filter:** Kauf-Signale nur, wenn der Kurs max. 1 ATR über der
  SMA10 (Pullback) bzw. dem Ausbruchslevel (Breakout) liegt — verhindert
  Signale, nachdem der Trend schon gelaufen ist.
- **Exit-Bestätigung:** Exit erst beim 2. Schluss unter der SMA20 oder wenn
  der Kurs klar (0,5 ATR) darunter schließt — verhindert Whipsaw.
- **Signal seit:** 1 = heute neu; höhere Zahl = Signal feuert schon länger,
  Einstieg entsprechend schlechter.
- **Zwei Ziele:** fixes 2R-Ziel und Struktur-Ziel (letztes Swing-Hoch).
- Signale basieren auf der **laufenden Tageskerze** und sind erst nach
  Börsenschluss endgültig.

**Backtest-Vergleich der Exit-Strategien** (alle Sektoren, 2 Jahre,
Survivorship-Bias beachten): Trailing-Stop 3×ATR ohne festes Ziel schlägt
das fixe 2R-Ziel deutlich (Ø ~0,7–1,1 R vs. ~0,25 R pro Trade, Profitfaktor
~2,2–2,8 vs. 1,5), weil Gewinner weiterlaufen dürfen. Das Struktur-Ziel
trifft öfter (56 %), bringt aber am wenigsten. Im 🧪 Backtest ist die
Exit-Strategie wählbar.

## OTE-Screener (Trend → Fib-Pullback → 4H-Bestätigung)

Die Ansicht 📐 setzt die klassische „Trade with the trend, buy the
pullback"-Methode um:

1. **Trendfilter:** Weekly-Schluss über steigendem 20-Wochen-Schnitt und
   Daily über SMA50 (SMA20 > SMA50). Nur diese Aktien erscheinen.
2. **Impuls-Erkennung:** letzter Aufwärts-Impuls (Swing-Tief → Swing-Hoch),
   mindestens 3 ATR groß; dazu der nächste Pivot-Support unter dem Kurs.
3. **OTE-Zone:** Rücklauf 62–79 % des Impulses (Sweet Spot 70,5 %).
   Über 88,6 % gilt das Setup als invalidiert.
4. **4H-Bestätigung:** 4H-Schluss über den Hochs der letzten sechs
   4H-Kerzen (Strukturbruch nach oben).
5. **Risiko:** Stop unter dem Swing-Tief (−¼ ATR), Ziel = altes Hoch,
   R:R wird berechnet.

Status-Stufen: ⏳ Rücklauf abwarten → 🟡 nähert sich → 🟠 im OTE (auf 4H
warten) → 🟢 ENTRY. Der Chart zeigt die Fib-Zone, Sweet Spot, Stop, Ziel
und Support. Dass oft **kein** Entry da ist, ist Absicht — Regel 5 des
Tutorials: warten, nicht erzwingen.

## Backtest

Die Ansicht 🧪 Backtest simuliert die Signal-Logik über 2–5 Jahre Historie:
Einstieg am Folgetag zur Eröffnung, Stop/2R-Ziel wie im Scanner, Exits über
Stop, Ziel, SMA20-Bruch oder Zeit-Exit (bei Stop und Ziel am selben Tag
zählt konservativ der Stop). Ergebnis: Trefferquote, Ø R, Profitfaktor,
Equity-Kurve und Aufschlüsselung nach Setup-Typ, Exit-Grund und Sektor.

**Achtung Survivorship-Bias:** Getestet werden die heutigen Ticker-Listen —
Aktien, die stark gelaufen sind, landen eher in der Watchlist. Die
Ergebnisse sind dadurch tendenziell zu optimistisch. Gebühren/Slippage
sind nicht eingerechnet.

## Hinweise

- Datenquelle: Yahoo Finance über `yfinance`, 15 Minuten Cache
  (Kursdaten sind ca. 15 min verzögert).
- Kurse werden in der Handelswährung des Tickers angezeigt (Spalte
  „Währung"); Kurs/Stop/Ziel zusätzlich in Euro, umgerechnet mit dem
  aktuellen Wechselkurs (EURUSD=X usw., 15 min Cache). Orders beim
  Broker laufen trotzdem in der Handelswährung — die Euro-Spalten
  dienen der Risiko- und Positionsgrößenrechnung.
- Kurse sind split-/dividendenbereinigt, damit die Overnight-Statistik
  nicht durch Dividendenabschläge verzerrt wird.
- Kein Handelssignal, keine Anlageberatung — ein Werkzeug zum Filtern
  von Kandidaten, nicht zum Timen von Einstiegen.
