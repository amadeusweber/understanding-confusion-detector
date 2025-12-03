import pandas as pd
from hampel import hampel
def annotate_blink(df:pd.DataFrame, col_au_blink:str, col_timestamp:str,
                   filter_window_size:int=17, filter_n_sigma:float=3.0) -> pd.DataFrame:
    '''
    Annotates blinks for a OpenFace processed video recording
    
    :param df: recording
    :type df: pd.DataFrame
    :param col_au_blink: Column name of blink Action Unit
    :type col_au_blink: str
    :param col_timestamp: Column name of frame timestamp
    :type col_timestamp: str
    :param filter_window_size: hample filter window size
    :type filter_window_size: int
    :param filter_n_sigma: hampel filter allowed deviation
    :type filter_n_sigma: float
    :return: df with additional columns for blink periods and intervals
    :rtype: DataFrame
    '''
    # sort by frame
    df = df.sort_values(col_timestamp).reset_index(drop=True)
    # filter raw blink data
    periods = pd.Series(hampel(
        data = df[col_au_blink],
        window_size=filter_window_size,
        n_sigma=filter_n_sigma
    ).filtered_data)
    # detect edges by differential
    period_edges = periods.diff() != 0
    period_starts = df[col_timestamp][period_edges].reset_index(drop=True)
    df['blink_state'] = periods
    df['blink_period_id'] = period_edges.cumsum()
    df['blink_period_begin'] = (df['blink_period_id'] - 1).map(period_starts)
    df['blink_period_end'] = df['blink_period_id'].map(period_starts)
    df['blink_period_time'] = df['blink_period_end'] - df['blink_period_begin']

    # detect falling edges only to identify full blink intervals
    interval_edges =  periods.diff().fillna(1) > 0
    interval_starts = df['timestamp'][interval_edges].reset_index(drop=True)
    df['blink_interval_id'] = interval_edges.cumsum()
    df['blink_interval_begin'] = (df['blink_interval_id'] - 1).map(interval_starts)
    df['blink_interval_end'] = (df['blink_interval_id']).map(interval_starts)
    df['blink_interval_time'] = df['blink_interval_end'] - df['blink_interval_begin']

    # return augmented df
    return df

def blink_processor(df:pd.DataFrame, col_au_blink:str, col_timestamp:str) -> pd.DataFrame:
    # decompose by participnat
    return pd.concat(
        # decompose by condition to obtain recording
        pd.concat(
            # add blink data
            annotate_blink(recording, col_au_blink, col_timestamp)
            for _, recording in participant.groupby('condition')
        )
        for _, participant in df.groupby('token')
    )