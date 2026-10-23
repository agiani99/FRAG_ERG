"""Summarise the PubChem substance source table (pubchem_sources.json) -> pubchem_sources.csv."""
import json, collections, csv
import pandas as pd
t = json.load(open('pubchem_sources.json', encoding='utf-8'))['Table']
cols = t['Columns']['Column']
df = pd.DataFrame([r['Cell'] for r in t['Row']], columns=cols)
df['n'] = pd.to_numeric(df['Live Substance Count'], errors='coerce').fillna(0).astype(int)
df = df.sort_values('n', ascending=False)
df[['Source Name', 'n', 'Source Category', 'Country', 'Source URL', 'Last Update']].to_csv('pubchem_sources.csv', index=False)
print(len(df), 'sources')
print(df.groupby('Source Category')['n'].agg(['count', 'sum']).sort_values('sum', ascending=False).head(15))
v = df[df['Source Category'].str.contains('Chemical Vendors', na=False)]
print(len(v), 'vendor sources,', v.n.sum(), 'substances')
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 500)
print(v[['Source Name', 'n', 'Country']].head(70).to_string(index=False))
kw = ['molport', 'emolecules', 'mcule', 'molmall', 'enamine', 'chemdiv', 'chembridge', 'vitas', 'zinc', 'chemspace', 'asinex', 'keyorganics', 'princeton', 'specs', 'aurora', 'life chem', 'otava', 'interbioscreen', 'sigma', 'tci', 'alfa', 'matrix', 'ambinter', 'namiki', 'bidepharm']
m = df[df['Source Name'].str.lower().str.contains('|'.join(kw))]
print(m[['Source Name', 'n', 'Source Category', 'Country']].to_string(index=False))
