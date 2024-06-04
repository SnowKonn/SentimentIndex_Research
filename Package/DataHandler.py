import pandas as pd
import datetime as dt
import numpy as np

class datahandler():
    def __init__(self, df: pd.DataFrame):
        assert 'date' in df.columns, "Column date should be included"
        self.df = df
        self.date_period_dict = {'w':['year_iso','week'],
                                 'm': ['year', 'month'],
                                 'y': ['year']}
    
    @staticmethod
    def get_logical_index(from_array: np.ndarray, to_array: np.ndarray, max_iter=100000):
        len_from = len(from_array)
        len_to = len(to_array)
        
        assert len_from == np.size(from_array), 'Input should be 1d array'
        assert len_to == np.size(to_array), 'Input should be 1d array'
        from_array = from_array.flatten()
        to_array = to_array.flatten()
        
        assert np.all(np.sort(from_array) == from_array), 'Input should be Asc ordered'
        assert np.all(np.sort(to_array) == to_array), 'Input should be Asc ordered'
        
        result_index_list = []
        
        from_i = 0
        to_i = 0
        loop_condition = True
        
        temp_i = 0
        
        while loop_condition:
            temp_i += 1
            if from_array[from_i] == to_array[to_i]:
                result_index_list.append(True)
                from_i += 1
                to_i += 1
            elif from_array[from_i] > to_array[to_i]:
                to_i += 1
            elif from_array[from_i] < to_array[to_i]:
                result_index_list.append(False)
                from_i += 1
            
            if (len_from <= from_i) or (len_to <= to_i):
                loop_condition = False
            elif temp_i > max_iter:
                break
               
                
        return np.array(result_index_list)
    

    
    def get_raw_df(self):
        return self.df.copy()
    
    def get_period_info_df(self):
        temp_df = self.df.copy()
        temp_df.set_index('date', drop=False, inplace=True)
        
        temp_df['year'] = temp_df.index.map(lambda x: x.year)
        temp_df['month'] = temp_df.index.map(lambda x: x.month)
        temp_df['year_iso'] = temp_df.index.map(
            lambda x: dt.date(x.year, x.month, x.day).isocalendar()[0]
            )
        temp_df['week'] = temp_df.index.map(
            lambda x: dt.date(x.year, x.month, x.day).isocalendar()[1]
            )
        
        return temp_df
    
    def _get_period_idx(self, period):
        
        week_idx = self.week_idx
        month_idx = self.month_idx
        year_idx = self.year_idx
        
        if period.lower() == 'w':
            period_idx = week_idx.values
        elif period.lower() == 'm':
            period_idx = month_idx.values
        elif period.lower() == 'y':
            period_idx = year_idx.values
        else:
            assert True, 'Please enter period among (w,m,y)'
        
        return period_idx

    
    def set_period_idx(self):
        temp_df = self.get_period_info_df()
        
        self.week_idx = temp_df.groupby(['year_iso', 'week'])[['date']].max()
        self.month_idx = temp_df.groupby(['year', 'month'])[['date']].max()
        self.year_idx = temp_df.groupby(['year'])[['date']].max()

    
    def get_data_by_period(self, period='w', columns=None):
        """
        df: Dataframe which contains date index, daily frequency data
        period: Period of data which would be return
        
        return df_
        """
        if columns is None:
            columns = self.df.columns
        
        temp_df = self.df.copy()
        temp_df.set_index('date', drop=False, inplace=True)

        if 'week_idx' not in self.__dict__.keys():
            self.set_period_idx()
        
        
        from_array = temp_df.index.values.T
        to_array = self._get_period_idx(period)
        logical_idx = self.get_logical_index(from_array,to_array)
        
        return self.df.loc[logical_idx, columns].copy()
    
    def get_period_sum_data(self, columns: list, period='w'):    
        if columns is None:
            columns = self.df.columns
        temp_df = self.get_period_info_df()
        df_sumwise = temp_df.\
            groupby(self.date_period_dict[period])[columns].\
            sum().\
            reset_index(drop=True)
            
        col_ordered_names = ['date'] + columns
        df_sumwise['date'] = self._get_period_idx(period)
        df_sumwise = df_sumwise[col_ordered_names]
        
        return df_sumwise
    

    
