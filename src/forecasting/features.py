import pandas as pd

def add_promo_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add promotional flags to the dataframe to help models capture holiday spikes.
    Specifically targets Black Friday / Cyber Monday and early December peak (weeks 47-50).
    """
    df = df.copy()
    
    # Extract ISO week if not already present
    if "week_of_year" not in df.columns:
        week_of_year = df["week"].dt.isocalendar().week.astype(int)
    else:
        week_of_year = df["week_of_year"]
        
    # Flag weeks 47 through 50 (late Nov through mid Dec)
    df["is_promo"] = ((week_of_year >= 47) & (week_of_year <= 50)).astype(int)
    
    return df
