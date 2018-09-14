# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:03:03 2018


A slim Pandas extension for having a quick Glance at datasets

Requires:
    pandas, numpy, matplotlib, tqdm
    
Todo:
    * Use multithreading in order to make it faster

@author: Gianfrancesco Angelini
"""



import sys
import os
import numpy as np
import re
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt



@pd.api.extensions.register_dataframe_accessor("glance")
class Glance( object ):
    '''
    
    Attributes:
        cols list[str]: Names of the columns of the inner DataFrame
        stats DataFrame: Statistics from glance()
    
    Example:
        >>>df = pd.read_csv( filename )
        >>>df = df.glance.glance()
        >>>df.glance.notnull()
    '''
    def __init__( self, pandas_obj ):
        self._obj  = pandas_obj
        self.cols  = list( self._obj.columns )
        self.stats = None # storage variable for glance()
        pd.set_option('display.max_columns', 18) # for DataFrame output
        
    ######### Helper functions   ##################
    # Yes, I will use this space soon or later...
    
    
    ######### Manual Exploration  #################
    
    def vexplore(self, cols = None, nrows=10):
        '''Vertical exploration
        
        It allows to slide along the records by a given step through a generator.
        
        Args:
            cols (list[str]): A list of columns to select.
            nrows (int): The numbers of rows to generate each time.
            
        Yields:
            DataFrame: The DataFrame chunk.
            
        Example:
            >>>df = pd.read_csv( filename )
            >>>agen = df.glance.vexplore( nrows=3 )
            >>>next(agen)
        '''
        nr = self._obj.count().max()
        if nr<nrows: nrows = nr
        if type(cols) == list:
            return ( self._obj[cols][x:x+nrows] for x in range(0, self._obj[cols].count().max()-nrows, nrows ) )
        return ( self._obj[x:x+nrows] for x in range(0, self._obj.count().max()-nrows, nrows ) )
    
    def hexplore(self, ncols=4, nrows=None):
        '''Horizontal exploration
        
        It allows to slide along the columns by a given columns step through a generator.
        
        Args:
            ncols (list[str]): Number of columns to generate each time.
            nrows (int): The numbers of first rows to generate.
        
        Yields:
            DataFrame: The DataFrame chunk.
        
        Example:
            >>>df = pd.read_csv( filename )
            >>>agen = df.glance.hexplore( ncols = 3 )
            >>>next(agen)
        '''
        nc = len( self._obj.columns )
        nr = self._obj.count().max()
        cols = list( self._obj.columns )
        if nc<ncols: ncols = nc
        if type(nrows) == int and nr<nrows: nrows = nr
        if type(nrows) == int:
            return ( self._obj[ cols[x:x+ncols] ][:nrows] for x in range(0, nc-ncols, ncols ) )
        return ( self._obj[ cols[x:x+ncols] ] for x in range(0, nc-ncols, ncols ) )
    
    
    ########## Checkers ##########################
    def can_be_date( self, col, over=0.5 ) -> list:
        '''Datetime values peeper
        
        Checks whether a column can be a date field, trying to convert it to
        such type and counting the fraction of accepted dates over the total.
        
        Args:
            col (str): The column to check over.
            over (float/int): fraction or amount of random rows over wich to perform the analysis.
        Returns:
            list[bool, float]: If True it's possibly a datetime column; the second element 
                states the confidence about the first element, as fraction of 
                convertible values over the column.
        
        Example:
            >>> df.glance.can_be_date('gebdat')
            [True, 99.99337899375418]
        
        @status: to verify
        '''
        tresh    = 0.8
        # Check compatible type
        if not( pd.api.types.infer_dtype( self._obj[col] ) == 'mixed' or pd.api.types.infer_dtype( self._obj[col] ) == 'string' or pd.api.types.is_datetime64_any_dtype( self._obj[col] )  ):
            return [ False, 0 ]
        # Check <over>
        if type(over) != None:
            if over>=1 and type(over) == int and self._obj[col].count() > over:
                serie = self._obj[col].sample( over )
            elif over<1 and type(over) == float:
                serie = self._obj[col].sample(frac=over)
        try:
            accepted = self._obj[ pd.to_datetime( self._obj[col], errors='coerce').notnull() ][col].count()
        except:
            return [False, 0]
        score    = accepted / self._obj[col].count()
        return [ score >= tresh, score*100 ]


    def can_be_email( self, col ) -> list:
        '''Email values peeper
        
        Checks whether a column can be a email field, looking for a email pattern
        in the values and counting the fraction of found emails over the total.
        
        Args:
            col (str): The column to check over.
            
        Returns:
            list[bool, float]: If True it's possibly a email column; the second element 
                states the confidence about the first element, as fraction of 
                found email values over the column.
        
        Example:
            >>> df.glance.can_be_email('email')
            [True, 99.24621939229365]
        '''
        tresh    = 0.8; 
        cond     = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        try:
            # Take all the notnull column values matching the condition and count them
            accepted = self._obj[ self._obj[col].str.match(cond)==True ][col].count()
        except:
            return [False, 0]
        score    = accepted / self._obj[col].count()
        return [ score >= tresh, score*100 ]
    
    
    ########### Overall operations ###################
    def notnull(self, col=None) -> pd.DataFrame:
        '''Notnull from Pandas
        
        Returns the notnull values of the Dataframe.
        
        Note:
            From the Pandas version you obtain a boolean Serie, here you obtain
            the entire Daframe, with the original column types, just pruned.
            
        Args:
            col (str): The name of a column to select as field to be notnull.
            
        Returns:
            Dataframe: The Dataframe containing only notnull values.
            
        Examples:
            >>> df.glance.notnull('city')
                                age      workclass  fnlwgt   
                0   39          State-gov   77516   Bachelors  
                1   50   Self-emp-not-inc   83311   Bachelors   
                2   38            Private  215646     HS-grad  
        '''
        if col!=None:
            return self._obj[ self._obj[col].notnull( ) ]
        return self._obj[ self._obj.notnull( ) ]
    
    
    def notnull_perc(self) -> pd.Series:
        '''Notnull percentage
        
        Returns the notnull percentage in columns of the Dataframe.
        
        Note:
            From the Pandas version you obtain a boolean Serie, here you obtain
            the entire Daframe, with the original column types, just pruned.
            
        Returns:
            Series: The percentages of not null values in columns.
            
        Examples:
            >>> df.glance.notnull_perc()
            age          1.0
            workclass    1.0
            fnlwgt       1.0
            dtype: float64
        '''
        return self._obj[ self._obj.notnull( ) ].count()  / self._obj.count().max()
    
    
    def isnull_count(self) -> pd.Series:
        '''Columns Null count
        
        Returns the null values count of the columns.
        
        Returns:
            Series: The count of the null values in columns.
        
        Examples:
            >>> df.glance.isnull_count()
            age          2
            workclass    5
            fnlwgt       0
            dtype: float64
        '''
        return self._obj.count().max() - self._obj[ self._obj.notnull( ) ].count()


    def cols_are_date(self, pbar=False) -> pd.DataFrame:
        '''Columns datetime seeker
        
        Returns the probability of the columns to host datetime values.
        
        Args:
            pbar (bool): If True shows a progress bar
        Returns:
            DataFrame: For each column of the original DataFrame you have two cols:
                the first is the boolean possibility of the the column to be a
                datetime field, the second is the confidence of such result,
                based on the frequency of the datetime convertible values present
                in the column.
        
        Examples:
            >>> df.glance.cols_are_date()
            ops        isDatetime   isDatetime conf
            Name          False                0
            email         False       0.00468187
            birthdate     True          99.9934
            city          False        0.0196618
        
        Todo:
            Make it faster! Maybe multithreading..
        '''
        res = { 'ops' : ['isDatetime', 'isDatetime conf'] }
        ncols = len(self.cols)
        with tqdm(total=100) as pbar: #tqdm progress bar, since this can be time spending
            for nc in range( ncols ):
                col = self.cols[nc]
                s_date   = self.can_be_date( col )
                res[col] = s_date
                if pbar:
                    pbar.update( 100*nc//ncols )
        return pd.DataFrame( res ).set_index('ops').transpose()
    
    
    def cols_are_email(self, pbar=False) -> pd.DataFrame:
        '''Columns email seeker
        
        Returns the probability of the columns to host email values.
        
        Args:
            pbar (bool): If True shows a progress bar
            
        Returns:
            DataFrame: For each column of the original DataFrame you have two cols:
                the first is the boolean possibility of the the column to be a
                datetime field, the second is the confidence of such result,
                based on the frequency of the datetime convertible values present
                in the column.
        
        Examples:
            >>> df.glance.cols_are_email()
            ops           isEmail     isEmail conf
            Name          False                0
            email         True         98.493494
            birthdate     False       0.00468187
            city          False        0.0196618
        
        '''
        res = { 'ops' : ['isEmail', 'isEmail conf'] }
        ncols = len(self.cols)
        with tqdm(total=100) as pbar: #tqdm progress bar, since this can be time spending
            for nc in range( ncols ):
                col = self.cols[nc]
                s_email  = self.can_be_email( col )
                res[col] = s_email
                if pbar:
                    pbar.update( 100*nc//ncols )
        return pd.DataFrame( res ).set_index('ops').transpose()
    
    
    ########## Compose the GLANCE ###########
    def glance(self) -> pd.DataFrame:
        '''Have a quick Glance on dataset!
        
        Builds a DataFrame describing some characteristics of the Dataset.
        
        Args:
            None
            
        Returns:
            DataFrame: Some stats to have quick hint about the dataset, organized
                along rows, for each column.
            
        Example:
            >>>df = pd.read_csv( filename )
            >>>df.glance.glance()
        '''
        isSomt = self.notnull().count()
        idxs   = list( self.notnull().count().keys() )  # freeze the index
        stats  = pd.DataFrame()
        stats['not_null'] = isSomt[idxs]
        stats['null']     = self._obj.count().max()-stats['not_null']
        stats['fullness'] = stats['not_null'] / self._obj.count().max()
        stats['dtype_origin']   = [  self._obj[col].dtype for col in idxs  ]
        stats['dtype_inferred'] = [  pd.api.types.infer_dtype( self._obj[col] ) for col in idxs  ]
        ## Heavy stuff here ##
        print('Scanning for dates...\n')
        stats = stats.join( self.cols_are_date(pbar = True) )
        print('Scanning for emails...\n')
        stats = stats.join( self.cols_are_email(pbar = True) )
        # Finally store the table
        self.stats = stats
        return stats
    
    
    
    
    
    
    ########## VISUALIZATION  ###############
    
    def glance_plot_nullness( self, sortby=None ) -> None:
        '''Plot Nullness
        
        Plots the fraction of null values, for each column, in stacked bars.
        
        Args:
            sortby (list[str]): ['not_null','null']
        
        Example:
            >>>df = pd.read_csv( filename )
            >>>df.glance.glance_plot_nullness()
        '''
        if self.stats == None:
            self.glance()
        if sortby!= None and len(sortby)>0:
            nan_bar = np.array( [v for k,v in self.stats.sort_values( by=sortby, ascending=False )['null'].items() ] )
            n_bar   = np.array( [v for k,v in self.stats.sort_values( by=sortby, ascending=False )['not_null'].items() ] )
        else:
            nan_bar = self.stats['null']
            n_bar   = self.stats['not_null']
        
        x =  list( stats.index )
        p1 = plt.bar( x, n_bar,   0.8 )
        p3 = plt.bar( x, nan_bar, 0.8, bottom = n_bar )
        
        plt.ylabel('n rows')
        plt.xlabel('fields')
        plt.title('Data Null values')
        plt.xticks( x, rotation = 60 )
        plt.legend( (p1[0], p3[0]), ('not null', 'null'))
        plt.show()
        return
    
    
##### Glance

