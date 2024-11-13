import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import time
import re
pd.options.mode.chained_assignment = None  # default='warn'
import concurrent.futures
import math
import numpy as np
import glob
import os
import multiprocessing

#we didn't import the below via hidden impors - that might cause problems, but let's see
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

import warnings
warnings.filterwarnings("ignore")


#Initial Sheet Loads
print("Process Started....")
def compileallcwrfiles():
    #grab and concat all admin cwr docs
    #strongly consider scrapping this if condition
    allFiles = [f for f in glob.glob(os.getcwd()+"/*.xlsx") if "CWR" in f]
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:        
        df = pd.read_excel(file_,index_col=[0], header=0, dtype= str,skiprows=[1])        
        list_.append(df)
    frame = pd.concat(list_,sort = False)
    frame.fillna('', inplace = True)
    frame = frame[pd.notnull(frame['Song Title'])]
    #this is supposed to skip wherethere isn't a title - seemed not to work, but I don't want to investigate if i needed it...solved below

    frame.replace({'’':"'",
                   ' {2,}':' ',
                   '¾':'3/4','½':'1/2','¼':'1/4','÷':'','#REF':'','#N/A':'',
                   "Š":"S",
                    "Œ":"CE",
                    "Ž":"Z",
                    "š":"s",
                    "œ":"oe",
                    "ž":"z",
                    "Ÿ":"Y",
                    "À":"A",
                    "Á":"A",
                    "Â":"A",
                    "Ã":"A",
                    "Ä":"A",
                    "Å":"A",
                    "Æ":"AE",
                    "Ç":"CE",
                    "È":"E",
                    "É":"E",
                    "Ê":"E",
                    "Ë":"E",
                    "Ì":"I",
                    "Í":"I",
                    "Î":"I",
                    "Ï":"I",
                    "Ð":"D",
                    "Ñ":"N",
                    "Ò":"O",
                    "Ó":"O",
                    "Ô":"O",
                    "Õ":"O",
                    "Ö":"O",
                    "×":"x",
                    "Ø":"O",
                    "Ù":"U",
                    "Ú":"U",
                    "Û":"U",
                    "Ü":"U",
                    "Ý":"Y",
                    "Þ":"d",
                    "ß":"B",
                    "à":"a",
                    "á":"a",
                    "â":"a",
                    "ã":"a",
                    "ä":"a",
                    "å":"a",
                    "æ":"ae",
                    "ç":"c",
                    "è":"e",
                    "é":"e",
                    "ê":"e",
                    "ë":"e",
                    "ì":"i",
                    "í":"i",
                    "î":"i",
                    "ï":"i",
                    "ð":"o",
                    "ñ":"n",
                    "ò":"o",
                    "ó":"o",
                    "ô":"o",
                    "õ":"o",
                    "ö":"o",
                    "÷":"",
                    "ø":"o",
                    "ù":"u",
                    "ú":"u",
                    "û":"u",
                    "ü":"u",
                    "ý":"v",
                    "þ":"b",
                    "ÿ":"y",
                    "ÿ":"y",
                    "\t":" ",
                    "\n":" ",
                   }, regex=True, inplace=True)    
    frame.replace({r'[^\x00-\x7F]+':' '}, regex=True, inplace=True)
    for column in frame.columns:#surely there's a better way to do this, but whatever.
        frame[column] = frame[column].apply(lambda x:x.strip())
    return frame
    
inputsheet = compileallcwrfiles()
#this might work....try a full run
inputsheet['Song Title'].replace('', np.nan, inplace=True)
inputsheet.dropna(subset=['Song Title'], inplace=True)

try:
    # Directly assign the specific file path to allFiles as a list
    #allFiles = [r"C:\Users\Dell_Office\Desktop\Projects\AdminWarning\Files\CPData.xlsx"]  # CP Data Path
    allFiles = pd.read_excel("CPData.xlsx", header=0, dtype=str)
    frame = pd.DataFrame()
    
    list_ = []
    
    if not list_:
        print("The list is empty.")
    else:
        print(f"List {list_}.")
        
    for file_ in allFiles:        
        df = pd.read_excel(file_, header=0,encoding = 'latin1', dtype= str)        
        list_.append(df)
    CPData = pd.concat(list_,sort = False)
    CPData.rename(columns={'Code': 'CPCODE',
                           'Title': 'CPTITLE',
                           'Composer': 'CPCOMPOSERS',
                           'Controlled Publishers': 'CPCLIENT',
                           'Main Artist': 'CPARTISTS',
                           'Song Clients': 'CPCLIENTONLYNOPUB'}, inplace=True)
    CPData['CPCLIENT']= CPData['CPCLIENT'] + CPData['CPCLIENTONLYNOPUB']
    CPData = CPData[['CPCODE','CPTITLE','CPCOMPOSERS','CPCLIENT','CPARTISTS','CPCLIENTONLYNOPUB']]
    CPData.fillna('', inplace = True)

except Exception as e:
    print(f"An error occurred: {e}")
    column_names = ['CPCODE','CPTITLE','CPCOMPOSERS','CPCLIENT','CPARTISTS','CPCLIENTONLYNOPUB']
    CPData = pd.DataFrame(columns = column_names)    
    CPData.loc[1] = ['','','','','','']

inputsheet['Input_Composer'] = inputsheet[['Composer 1 First Name',#controlled
                                           'Composer 1 Surname',
                                           'Composer 2 First Name',
                                           'Composer 2 Surname',
                                           'Composer 3 First Name',
                                           'Composer 3 Surname',
                                           'Composer 4 First Name',
                                           'Composer 4 Surname',
                                           'Composer 5 First Name',
                                           'Composer 5 Surname',
                                           'Composer 6 First Name',
                                           'Composer 6 Surname',
                                           'Composer 7 First Name',
                                           'Composer 7 Surname',
                                           'Composer 8 First Name',
                                           'Composer 8 Surname',
                                           'Composer 9 First Name',
                                           'Composer 9 Surname',
                                           #noncontrolled
                                           'Composer 40 First Name',
                                           'Composer 40 Surname',
                                           'Composer 41 First Name',
                                           'Composer 41 Surname',
                                           'Composer 42 First Name',
                                           'Composer 42 Surname',
                                           'Composer 43 First Name',
                                           'Composer 43 Surname',
                                           'Composer 44 First Name',
                                           'Composer 44 Surname',
                                           'Composer 45 First Name',
                                           'Composer 45 Surname',
                                           'Composer 46 First Name',
                                           'Composer 46 Surname',
                                           'Composer 47 First Name',
                                           'Composer 47 Surname',
                                           'Composer 48 First Name',
                                           'Composer 48 Surname',
                                           'Composer 49 First Name',
                                           'Composer 49 Surname',
                                           'Composer 50 First Name',
                                           'Composer 50 Surname',
                                           #11-24 is 
                                           'Composer 11 First Name',
                                           'Composer 11 Surname',
                                           'Composer 12 First Name',
                                           'Composer 12 Surname',
                                           'Composer 13 First Name',
                                           'Composer 13 Surname',
                                           'Composer 14 First Name',
                                           'Composer 14 Surname',
                                           'Composer 15 First Name',
                                           'Composer 15 Surname',
                                           'Composer 16 First Name',
                                           'Composer 16 Surname',
                                           'Composer 17 First Name',
                                           'Composer 17 Surname',
                                           'Composer 18 First Name',
                                           'Composer 18 Surname',
                                           'Composer 19 First Name',
                                           'Composer 19 Surname',
                                           'Composer 20 First Name',
                                           'Composer 20 Surname',
                                           'Composer 21 First Name',
                                           'Composer 21 Surname',
                                           'Composer 22 First Name',
                                           'Composer 22 Surname',
                                           'Composer 23 First Name',
                                           'Composer 23 Surname',
                                           'Composer 24 First Name',
                                           'Composer 24 Surname',                                      
                                           ]].agg(' '.join, axis=1)


#alright....so what if we induce a non-asci character as a split, then apply a function to split? ya kow what....like the CP data could have dups....
def inputpubreduce(Input_Publisher):
    publist = Input_Publisher.split('Œ')
    publist = list(dict.fromkeys(publist))
    #now we're getting an error where it's pulling the first row
    stringlist = ' '.join(publist)
    return stringlist


def totalimportedsummations(Input):
    publist = Input.split('Œ')
    publist = list(dict.fromkeys(publist))
    stringlist = '|'.join(publist)
    t = stringlist.replace('Œ',' ').lstrip('|').strip().rstrip('|')
    return t

def totalimportedcompsummations(Input):
    publist = Input.split('Œ')
    publist = list(dict.fromkeys(publist))
    stringlist = '|'.join(publist)
    t = stringlist.replace('Œ',' ').lstrip('|').strip().rstrip('|')
    return t

inputsheet['Total Imported Clients'] = inputsheet[['Composer 1 Associated Client',
                                           'Composer 2 Associated Client',
                                           'Composer 3 Associated Client',
                                           'Composer 4 Associated Client',
                                           'Composer 5 Associated Client',
                                           'Composer 6 Associated Client',
                                           'Composer 7 Associated Client',
                                           'Composer 8 Associated Client',
                                           'Composer 9 Associated Client',                                         
                                           ]].agg('Œ'.join, axis=1)

inputsheet['Total Imported Controlled Publishers'] = inputsheet[['Composer 1 Linked Publisher',
                                           'Composer 2 Linked Publisher',
                                           'Composer 3 Linked Publisher',
                                           'Composer 4 Linked Publisher',
                                           'Composer 5 Linked Publisher',
                                           'Composer 6 Linked Publisher',
                                           'Composer 7 Linked Publisher',
                                           'Composer 8 Linked Publisher',
                                           'Composer 9 Linked Publisher',                                       
                                           ]].agg('Œ'.join, axis=1)

for x in range(1,10):
    x = str(x)
    inputsheet['Composer '+x+' Full'] = inputsheet[['Composer '+x+' First Name',
                                           'Composer '+x+' Surname']].agg(' '.join, axis=1)
    
#there's still a dup reduce, but this might make it better
inputsheet['Total Imported Controlled Composers'] = inputsheet[['Composer 1 Full',
                                            'Composer 2 Full',
                                            'Composer 3 Full',
                                            'Composer 4 Full',
                                            'Composer 5 Full',
                                            'Composer 6 Full',
                                            'Composer 7 Full',
                                            'Composer 8 Full',
                                            'Composer 9 Full'                                                                 
                                           ]].agg('Œ'.join, axis=1)
    
#this is for the fuzzy - you want both here
inputsheet['Input_Publisher'] = inputsheet[['Composer 1 Linked Publisher',
                                           'Composer 1 Associated Client',
                                           'Composer 2 Linked Publisher',
                                           'Composer 2 Associated Client',
                                           'Composer 3 Linked Publisher',
                                           'Composer 3 Associated Client',
                                           'Composer 4 Linked Publisher',
                                           'Composer 4 Associated Client',
                                           'Composer 5 Linked Publisher',
                                           'Composer 5 Associated Client',
                                           'Composer 6 Linked Publisher',
                                           'Composer 6 Associated Client',
                                           'Composer 7 Linked Publisher',
                                           'Composer 7 Associated Client',
                                           'Composer 8 Linked Publisher',
                                           'Composer 8 Associated Client',
                                           'Composer 9 Linked Publisher',
                                           'Composer 9 Associated Client',
                                           'Composer 40 Linked Publisher',
                                           'Composer 41 Linked Publisher',
                                           'Composer 42 Linked Publisher',
                                           'Composer 43 Linked Publisher',
                                           'Composer 44 Linked Publisher',
                                           'Composer 45 Linked Publisher',
                                           'Composer 46 Linked Publisher',
                                           'Composer 47 Linked Publisher',
                                           'Composer 48 Linked Publisher',
                                           'Composer 49 Linked Publisher',
                                           'Composer 50 Linked Publisher',                                           
                                           ]].agg('Œ'.join, axis=1)

inputsheet['Total Imported Clients'] = inputsheet['Total Imported Clients'].apply(totalimportedsummations)
inputsheet['Total Imported Controlled Publishers'] = inputsheet['Total Imported Controlled Publishers'].apply(totalimportedsummations)
inputsheet['Total Imported Controlled Composers'] = inputsheet['Total Imported Controlled Composers'].apply(totalimportedcompsummations)

inputsheet['Input_Publisher'] = inputsheet['Input_Publisher'].apply(inputpubreduce)




#so these client shares things....like honestly....to generate buyin we might have to default to doing lookups from the master meta...it sucks and the fuzzies are better
#columns jl to kd

#jump here


#ke - pub sum - relys on stuff that is generated later....looks like we start with

#anything before JL can be independently derived from the standardized input sheet


def controlledcomposerscapacity(data):
    #data[0],data[1]   
    if (data[0] != '') or (data[1] !=''):
        return 'CA'

def controlledcomposersMOMC(data):
    #data[0],data[1]
    #these try and except blocks are stupid as fuck - amend that
    try:
        if (data[0] != '') or (data[1] !=''):
            return 0
    except:
        pass

def controlledpublishersMOMC(data):
    #data[0],data[1]
    #these try and except blocks are stupid as fuck - amend that
    try:
        if (data[0] != '') or (data[1] !=''):
            return 0
    except:
        pass


def controlledcomposersshare(data):
    #data[0],data[1]
    #these try and except blocks are stupid as fuck - amend that
    try:
        if (data != '') or (data != '0'):
            #x = int(data)/2
            return int(data)/2
        if (data != '0'):
            return 0
    except:
        pass


def composer40pluscontrolled(data):
    if (data[0] == '') and (data [1] == ''):
        pass
    else:
        return "N"
        

def comp11linkedpub(data):
    if (data[0] == '') and (data [1] == ''):
        pass
    else:
        return "Publisher Unknown"
#jumphere
for x in range(1,25):
    x = str(x)
    if x == '10':
        pass
    else:
        inputsheet['Composer '+x+' Capacity'] = inputsheet[['Composer '+x+' First Name','Composer '+x+' Surname']].apply(controlledcomposerscapacity,axis=1)    
        inputsheet['Composer '+x+' MO Share'] = inputsheet['Composer '+x+' Capacity'].apply(controlledcomposersMOMC)
        inputsheet['Composer '+x+' MC Share'] = inputsheet['Composer '+x+' MO Share']
        

for x in range(40,51):
    x = str(x)
    inputsheet['Composer '+x+' Capacity'] = inputsheet[['Composer '+x+' First Name','Composer '+x+' Surname']].apply(controlledcomposerscapacity,axis=1)    
    inputsheet['Composer '+x+' MO Share'] = inputsheet['Composer '+x+' Capacity'].apply(controlledcomposersMOMC)
    inputsheet['Composer '+x+' MC Share'] = inputsheet['Composer '+x+' MO Share']    
    inputsheet['Composer '+x+' Controlled'] = inputsheet[['Composer '+x+' First Name','Composer '+x+' Surname']].apply(composer40pluscontrolled,axis=1) 
    
for x in range(1,10):
    x = str(x)
    inputsheet['Composer '+x+' Share (Total = 100)'] = pd.to_numeric(inputsheet['Composer '+x+' Share (Total = 100)'], errors='coerce')
    inputsheet['Composer '+x+' PO Share'] = np.where(inputsheet['Composer '+x+' Share (Total = 100)']=='','',inputsheet['Composer '+x+' Share (Total = 100)']/2)
    inputsheet['Composer '+x+' PC Share'] = inputsheet['Composer '+x+' PO Share']


for x in range(40,51):
    x = str(x)
    inputsheet['Composer '+x+' PO Share'] = inputsheet['Composer '+x+' Share (Total composer share = 100)'].apply(controlledcomposersshare)
    inputsheet['Composer '+x+' PC Share'] = inputsheet['Composer '+x+' PO Share']


#composer shares
#jumphere - Composer (Other) Share (Simple; No Import)
#composer PO shares 11-24 rely on this calculation, at least in danny's version = Composer (Other) Share (Simple; No Import)
#OK - so instead of


def composerothersharefound(data):
    i = 0
    for x in range(0,14):
        if (data[x] == 'N') or (data[x] == 'Y'):# and (data[2] == '') and (data[3] == ''):
            i=i+1
    #print(i)
    return i

for col in  inputsheet[["Composer 1 Share (Total = 100)",
                        "Composer 2 Share (Total = 100)",
                        "Composer 3 Share (Total = 100)",
                        "Composer 4 Share (Total = 100)",
                        "Composer 5 Share (Total = 100)",
                        "Composer 6 Share (Total = 100)",
                        "Composer 7 Share (Total = 100)",
                        "Composer 8 Share (Total = 100)",
                        "Composer 9 Share (Total = 100)",
                        "Composer 40 Share (Total composer share = 100)",
                         "Composer 41 Share (Total composer share = 100)",
                         "Composer 42 Share (Total composer share = 100)",
                         "Composer 43 Share (Total composer share = 100)",
                          "Composer 44 Share (Total composer share = 100)",
                          "Composer 45 Share (Total composer share = 100)",
                          "Composer 46 Share (Total composer share = 100)",
                          "Composer 47 Share (Total composer share = 100)",
                          "Composer 48 Share (Total composer share = 100)",
                          "Composer 49 Share (Total composer share = 100)",
                          "Composer 50 Share (Total composer share = 100)",                        
                        ]]:
    inputsheet[col] = pd.to_numeric(inputsheet[col], errors='coerce')
    
#alright....so this is for step 2 - we need to bring the convert to numerics up here
inputsheet['COMP SUMshort'] = inputsheet[["Composer 1 Share (Total = 100)",
                                     "Composer 2 Share (Total = 100)",
                                     "Composer 3 Share (Total = 100)",
                                     "Composer 4 Share (Total = 100)",
                                     "Composer 5 Share (Total = 100)",
                                     "Composer 6 Share (Total = 100)",
                                     "Composer 7 Share (Total = 100)",
                                     "Composer 8 Share (Total = 100)",
                                     "Composer 9 Share (Total = 100)",
                                     "Composer 40 Share (Total composer share = 100)",
                                     "Composer 41 Share (Total composer share = 100)",
                                     "Composer 42 Share (Total composer share = 100)",
                                     "Composer 43 Share (Total composer share = 100)",
                                      "Composer 44 Share (Total composer share = 100)",
                                      "Composer 45 Share (Total composer share = 100)",
                                      "Composer 46 Share (Total composer share = 100)",
                                      "Composer 47 Share (Total composer share = 100)",
                                      "Composer 48 Share (Total composer share = 100)",
                                      "Composer 49 Share (Total composer share = 100)",
                                      "Composer 50 Share (Total composer share = 100)",
                                    ]].sum(axis=1)

#now we need to switch the PO share to rely on the column above
for x in range(11,25):
    x = str(x)
    inputsheet['Composer '+x+' Controlled'] = inputsheet[['Composer '+x+' First Name','Composer '+x+' Surname']].apply(composer40pluscontrolled,axis=1)
    inputsheet['Composer '+x+' Linked Publisher'] = inputsheet[['Composer '+x+' First Name','Composer '+x+' Surname']].apply(comp11linkedpub,axis=1)    

inputsheet['Composer (Other) Share (Simple; No Import)'] = inputsheet[["Composer 11 Controlled",
                                                                        "Composer 12 Controlled",
                                                                        "Composer 13 Controlled",
                                                                        "Composer 14 Controlled",
                                                                        "Composer 15 Controlled",
                                                                        "Composer 16 Controlled",
                                                                        "Composer 17 Controlled",
                                                                        "Composer 18 Controlled",
                                                                        "Composer 19 Controlled",
                                                                        "Composer 20 Controlled",
                                                                        "Composer 21 Controlled",
                                                                        "Composer 22 Controlled",
                                                                        "Composer 23 Controlled",
                                                                        "Composer 24 Controlled"]].apply(composerothersharefound, axis=1)#.apply(lambda x:x)#.notnull().all(axis=1)#.apply(lambda x:x) maybe instead?


def composerothersharefoundsteptwo(data):
    #print(data)
    if data[0] > 0:
        return (100-data[1])/data[0]
    else:
        return ''

def prorataassigncorrect(data):
    #print(data)
    if data[1] == 0:
        return data[0]
    else:
        return ''
        
inputsheet['Composer (Other) Share (Simple; No Import)'] = inputsheet[['Composer (Other) Share (Simple; No Import)','COMP SUMshort']].apply(composerothersharefoundsteptwo,axis=1)

for x in range(11,25):
    x = str(x)
    #inputsheet['Composer '+x+' PO Share'] = inputsheet[['Composer '+x+' First Name','Composer '+x+' Surname']].apply(controlledcomposersMOMC)
    #inputsheet['Composer '+x+' PO Share'] = #inputsheet['Composer (Other) Share (Simple; No Import)']#ok - so there's one more element of danny's formula here -
    #the error we're having is that it's applying the pro-ration to every single one, rather than assigning to only the relevant parties
    inputsheet['Composer '+x+' PO Share'] = inputsheet[['Composer (Other) Share (Simple; No Import)','Composer '+x+' MO Share']].apply(prorataassigncorrect,axis=1)
    inputsheet['Composer '+x+' PC Share'] = inputsheet['Composer '+x+' PO Share']

#jumphere

def pubcapacity(data):
    if data == '':
        pass
    else:
        return 'AM'

def pubMo(data):
    if data == '':
        pass
    else:
        return 0

def pubMc(data):
    if data[0] == '':
        #print(data[0])
        pass
    else:
        #print(data[1])#ok it is data1 that's need
        #float(data[1])
        data[1]

def pubPo(data):
    if data[0] == '':
        pass
    else:
        #float(data[1])
        data[1]

#Publisher Shares - the mc, PO, and PC shares are messed up
for x in range(11,20):
    x = str(x)
    inputsheet['Publisher '+x+' Name'] = inputsheet['Publisher '+str(int(x)-10)+' Linked Publisher']    
    inputsheet['Publisher '+x+' Capacity'] = inputsheet['Publisher '+str(int(x)-10)+' Linked Publisher'].apply(pubcapacity)
    inputsheet['Publisher '+x+' MO Share'] = inputsheet['Publisher '+str(int(x)-10)+' Linked Publisher'].apply(pubMo)
    inputsheet['Publisher '+x+' MC Share'] = inputsheet[['Publisher '+str(int(x)-10)+' Linked Publisher','Publisher '+str(int(x)-10)+' MO Share']].apply(pubMc,axis=1)
    inputsheet['Publisher '+x+' PO Share'] = np.where(inputsheet['Publisher '+str(int(x)-10)+' Linked Publisher'] == '', 0, '')
    inputsheet['Publisher '+x+' PC Share'] = np.where(['Publisher '+str(int(x)-10)+' PC Share']=='', '',inputsheet['Publisher '+x+' MC Share']/2)
    
for x in range(1,10):
    x = str(x)
    inputsheet['Publisher '+x+' Capacity'] = np.where(inputsheet['Publisher '+x+' Name'] == '','','OP')
    inputsheet['Publisher '+x+' MC Share'] = inputsheet['Publisher '+x+' MO Share']
    inputsheet['Publisher '+x+' PO Share'] = np.where(inputsheet['Publisher '+x+' Name'] == '','',inputsheet['Composer '+x+' Share (Total = 100)']/2)
    inputsheet['Publisher '+x+' PC Share'] = np.where(inputsheet['Publisher '+x+' Linked Publisher'] == '',inputsheet['Publisher '+x+' PO Share'],0)

#alright - so publisher 11 through 24 don't really exist in the admin cwr - we can leave that shit be if we really want, but in the import 

for x in range(40,51):
    x = str(x)
    #new code
    #Composer 40 Share (Total composer share = 100)
    #Composer 40 Linked Publisher
    #inputsheet['Composer '+x+' Linked Publisher'] = np.where((inputsheet['Composer '+x+' Share (Total composer share = 100)'] != '') & (inputsheet['Composer '+x+' Linked Publisher'].isna()),'Publisher Unknown',inputsheet['Publisher '+x+' Name'])
    inputsheet['Composer '+x+' Linked Publisher'] = np.where((~inputsheet['Composer '+x+' Share (Total composer share = 100)'].isna()) & (inputsheet['Composer '+x+' Linked Publisher'] == ''),'Publisher Unknown',inputsheet['Composer '+x+' Linked Publisher'])    
    inputsheet['Publisher '+x+' Name'] = inputsheet['Composer '+x+' Linked Publisher']
    inputsheet['Publisher '+x+' MO Share'] = inputsheet['Composer '+x+' Share (Total composer share = 100)']
    #newcode above
    inputsheet['Publisher '+x+' Controlled'] = np.where(inputsheet['Publisher '+x+' Name'] == '','','N')
    inputsheet['Publisher '+x+' Capacity'] = np.where(inputsheet['Publisher '+x+' Controlled'] == '','','OP')
    inputsheet['Publisher '+x+' MC Share'] = np.where(inputsheet['Publisher '+x+' Controlled'] == '','','OP')
    inputsheet['Publisher '+x+' PO Share'] = np.where(inputsheet['Publisher '+x+' Controlled'] == '','',inputsheet['Composer '+x+' Share (Total composer share = 100)']/2)
    inputsheet['Publisher '+x+' PC Share'] = inputsheet['Publisher '+x+' PO Share']



#pubsum and other addition functions
for col in  inputsheet[["Publisher 1 MO Share",
                        "Publisher 2 MO Share",
                        "Publisher 3 MO Share",
                        "Publisher 4 MO Share",
                        "Publisher 5 MO Share",
                        "Publisher 6 MO Share",
                        "Publisher 7 MO Share",
                        "Publisher 8 MO Share",
                        "Publisher 9 MO Share",
                        "Publisher 40 MO Share",
                        "Publisher 41 MO Share",
                        "Publisher 42 MO Share",
                        "Publisher 43 MO Share",
                        "Publisher 44 MO Share",
                        "Publisher 45 MO Share",
                        "Publisher 46 MO Share",
                        "Publisher 47 MO Share",
                        "Publisher 48 MO Share",
                        "Publisher 49 MO Share",
                        "Publisher 50 MO Share",
                        "Publisher 11 PO Share",
                        "Publisher 12 PO Share",
                        "Publisher 13 PO Share",
                        "Publisher 14 PO Share",
                        "Publisher 15 PO Share",
                        "Publisher 16 PO Share",
                        "Publisher 17 PO Share",
                        "Publisher 18 PO Share",
                        "Publisher 19 PO Share",
                        "Entered Writer Share Total"
                        ]]:
    inputsheet[col] = pd.to_numeric(inputsheet[col], errors='coerce')

    
inputsheet['PUB SUM'] = inputsheet[["Publisher 1 MO Share",
                                    "Publisher 2 MO Share",
                                    "Publisher 3 MO Share",
                                    "Publisher 4 MO Share",
                                    "Publisher 5 MO Share",
                                    "Publisher 6 MO Share",
                                    "Publisher 7 MO Share",
                                    "Publisher 8 MO Share",
                                    "Publisher 9 MO Share",
                                    "Publisher 40 MO Share",
                                    "Publisher 41 MO Share",
                                    "Publisher 42 MO Share",
                                    "Publisher 43 MO Share",
                                    "Publisher 44 MO Share",
                                    "Publisher 45 MO Share",
                                    "Publisher 46 MO Share",
                                    "Publisher 47 MO Share",
                                    "Publisher 48 MO Share",
                                    "Publisher 49 MO Share",
                                    "Publisher 50 MO Share",
                                    "Publisher 11 PO Share",
                                    "Publisher 12 PO Share",
                                    "Publisher 13 PO Share",
                                    "Publisher 14 PO Share",
                                    "Publisher 15 PO Share",
                                    "Publisher 16 PO Share",
                                    "Publisher 17 PO Share",
                                    "Publisher 18 PO Share",
                                    "Publisher 19 PO Share",
                                    ]].sum(axis=1)

#for x in range(21,22):#this is the 'unknown publisher' - it's the catchall for extra shares
    #x = str(x)
# Convert relevant columns to numeric, coercing errors to NaN

columns_to_convert = [
    "Composer 1 Share (Total = 100)",
    "Composer 2 Share (Total = 100)",
    "Composer 3 Share (Total = 100)",
    "Composer 4 Share (Total = 100)",
    "Composer 5 Share (Total = 100)",
    "Composer 6 Share (Total = 100)",
    "Composer 7 Share (Total = 100)",
    "Composer 8 Share (Total = 100)",
    "Composer 9 Share (Total = 100)",
    "Composer 40 Share (Total composer share = 100)",
    "Composer 41 Share (Total composer share = 100)",
    "Composer 42 Share (Total composer share = 100)",
    "Composer 43 Share (Total composer share = 100)",
    "Composer 44 Share (Total composer share = 100)",
    "Composer 45 Share (Total composer share = 100)",
    "Composer 46 Share (Total composer share = 100)",
    "Composer 47 Share (Total composer share = 100)",
    "Composer 48 Share (Total composer share = 100)",
    "Composer 49 Share (Total composer share = 100)",
    "Composer 50 Share (Total composer share = 100)",
    "Composer 11 PO Share",
    "Composer 12 PO Share",
    "Composer 13 PO Share",
    "Composer 14 PO Share",
    "Composer 15 PO Share",
    "Composer 16 PO Share",
    "Composer 17 PO Share",
    "Composer 18 PO Share",
    "Composer 19 PO Share",
    "Composer 20 PO Share",
    "Composer 21 PO Share",
    "Composer 22 PO Share",
    "Composer 23 PO Share",
    "Composer 24 PO Share",
]

for col in columns_to_convert:
    inputsheet[col] = pd.to_numeric(inputsheet[col], errors='coerce')

# Now perform the sum
inputsheet['COMP SUM'] = inputsheet[columns_to_convert].sum(axis=1)
#inputsheet['Publisher 21 Name'] = np.where(inputsheet['PUB SUM'] < 99.97, '','Publisher Unknown')
inputsheet['Publisher 21 Name'] = np.where(inputsheet['PUB SUM'] < 99.97, 'Publisher Unknown','')
inputsheet['Publisher 21 Controlled'] = np.where(inputsheet['PUB SUM'] >= 99.999999999999, '','No')
inputsheet['Publisher 21 Capacity'] = np.where(inputsheet['PUB SUM'] >= 99.999999999999, '','OP')

inputsheet['Publisher 21 MO Share'] = np.where(inputsheet['PUB SUM'] >= 99.999999999999, '',(100 - inputsheet['PUB SUM']))
inputsheet['Publisher 21 MC Share'] = inputsheet['Publisher 21 MO Share']

for col in  inputsheet[["Publisher 21 MO Share",
                        "Publisher 21 MC Share",
                        ]]:
    inputsheet[col] = pd.to_numeric(inputsheet[col], errors='coerce')

inputsheet['Publisher 21 PO Share'] = np.where(inputsheet['PUB SUM'] >= 99.999999999999, '',inputsheet['Publisher 21 MO Share']/2)#this may have a problem
inputsheet['Publisher 21 PC Share'] = inputsheet['Publisher 21 PO Share']


inputsheet['COMP SUM'] = inputsheet[["Composer 1 Share (Total = 100)",
                                     "Composer 2 Share (Total = 100)",
                                     "Composer 3 Share (Total = 100)",
                                     "Composer 4 Share (Total = 100)",
                                     "Composer 5 Share (Total = 100)",
                                     "Composer 6 Share (Total = 100)",
                                     "Composer 7 Share (Total = 100)",
                                     "Composer 8 Share (Total = 100)",
                                     "Composer 9 Share (Total = 100)",
                                     "Composer 40 Share (Total composer share = 100)",
                                     "Composer 41 Share (Total composer share = 100)",
                                     "Composer 42 Share (Total composer share = 100)",
                                     "Composer 43 Share (Total composer share = 100)",
                                      "Composer 44 Share (Total composer share = 100)",
                                      "Composer 45 Share (Total composer share = 100)",
                                      "Composer 46 Share (Total composer share = 100)",
                                      "Composer 47 Share (Total composer share = 100)",
                                      "Composer 48 Share (Total composer share = 100)",
                                      "Composer 49 Share (Total composer share = 100)",
                                      "Composer 50 Share (Total composer share = 100)",
                                     "Composer 11 PO Share",
                                     "Composer 12 PO Share",
                                     "Composer 13 PO Share",
                                     "Composer 14 PO Share",
                                     "Composer 15 PO Share",
                                     "Composer 16 PO Share",
                                     "Composer 17 PO Share",
                                     "Composer 18 PO Share",
                                     "Composer 19 PO Share",
                                     "Composer 20 PO Share",
                                     "Composer 21 PO Share",
                                     "Composer 22 PO Share",
                                     "Composer 23 PO Share",
                                     "Composer 24 PO Share",
                                    ]].sum(axis=1)


#index fixes and ISRC Dup Counter
inputsheet['CWRIndex'] = inputsheet.index
inputsheet.index = [x for x in range(1, len(inputsheet.values)+1)]
inputsheet.index.name = 'id'
#inputsheet.to_excel('InputSheetindexcheck.xlsx')
#inputsheet['isrccount'] = inputsheet.groupby(['Recording 1 ISRC'])['Song Title'].transform('count')#is that dangerously combining same titled works?
inputsheet['isrccount'] = inputsheet.groupby(['Recording 1 ISRC'])['CWRIndex'].transform('count')
inputsheet['CP Song Code Count'] = inputsheet.groupby(['CP Song Code'])['CWRIndex'].transform('count')


def cpsongcoderegex(data):
    if "_" in str(data):#could've made this CWR but I don't think people follow the naming convention
        return ''
    if "DNI//" in str(data):
        return str(data).replace("DNI//","")

    else:
        return str(re.sub(r'\([^)]*\)', '', data)).strip()
    
inputsheet['CP Song CodeEDITREGEX'] = inputsheet['CP Song Code'].apply(cpsongcoderegex)


x = [col for col in inputsheet if 'Share' in col]
for col in  inputsheet[x]:
    inputsheet[col] = pd.to_numeric(inputsheet[col], errors='coerce')
inputsheet.fillna('')

inputsheet['Imported Controlled Share'] = inputsheet[['Composer 1 Share (Total = 100)',
                                                      'Composer 2 Share (Total = 100)',
                                                      'Composer 3 Share (Total = 100)',
                                                      'Composer 4 Share (Total = 100)',
                                                      'Composer 5 Share (Total = 100)',
                                                      'Composer 6 Share (Total = 100)',
                                                      'Composer 7 Share (Total = 100)',
                                                      'Composer 8 Share (Total = 100)',
                                                      'Composer 9 Share (Total = 100)',
                                                      ]].sum(axis=1)

ISRCList = pd.read_csv(r"C:\Users\Dell_Office\Desktop\Projects\AdminWarning\Files\MDH Song Recording.txt", encoding = 'latin1', delimiter=",", dtype= str) # MDH file path

ISRCList.fillna('', inplace = True)
ISRCList['ISRC'] = ISRCList['ISRC'].apply(lambda x:str(x).replace("-",""))
                           
if __name__ == '__main__':

    allresults = inputsheet
    def warnings(allresults):    
         warning = []
         if (allresults['Imported Controlled Share'] == 0) and (allresults['CP Song Code'] ==''):
             warning.append("IP Info Left Blank and no DNI applied")
         if allresults['Recording 1 ISRC'] != '':
             if allresults['isrccount'] > 1:
                 warning.append("Duplicate ISRC Present In Sheet")
             if len(ISRCList.loc[ISRCList.ISRC == allresults['Recording 1 ISRC']]) > 0:
                 x = [i for i in ISRCList.loc[ISRCList.ISRC == allresults['Recording 1 ISRC']].Song_Code.values]
                 warning.append("ISRC Present in Existing CP Codes: "+str(x))
         if "DNI" in allresults['CP Song Code'] and allresults['Input_Composer'] != '                                                                   ':#make sure that's correct
             #this probably has a space in there...gonna add one and see if it checks
             warning.append("DNI Used Incorrectly - do you want writer info amended CP?")
         #if "DNI" not in allresults['CP Song Code'] and allresults['CP Song Code'] != '':
         if "DNI" not in allresults['CP Song Code'] and allresults['CP Song Code'] != '' and "CWR" not in allresults['CP Song Code']:             
             if allresults['CP Song Code Count'] > 1:
                 warning.append("CP Song Code Applied More Than Once")
        
         for x in range(1,10):#so this premise works...seems like warning conditions are a little funky though
             x = str(x)
             if (allresults['Composer '+x+' First Name'] != '' or allresults['Composer '+x+' Surname'] != '') and (allresults['Composer '+x+' Linked Publisher'] == '' or allresults['Composer '+x+' Associated Client'] == '' or allresults['Composer '+x+' Share (Total = 100)'] < 0.0000000000000001 or allresults['Publisher '+x+' Name'] == ''):                 
                  warning.append("Composer "+x+" Missing Data")

             #if (~allresults['Composer '+x+' Share (Total = 100)'].isnull()) and (allresults['Composer '+x+' First Name'] == '' and allresults['Composer '+x+' Surname'] == ''):
             if (allresults['Composer '+x+' Share (Total = 100)']>0) and (allresults['Composer '+x+' First Name'] == '' and allresults['Composer '+x+' Surname'] == ''):                 
                 warning.append('Composer '+x+' Share provided with no data')
                 
             if (allresults['Publisher '+x+' US Society'] == '' or allresults['Publisher '+x+' Affiliation'] == '') or (allresults['Publisher '+x+' US Society'] not in {'ASCAP','BMI','SESAC'}):                 
                 if (allresults['Composer '+x+' First Name'] == '' and allresults['Composer '+x+' Surname'] == ''):
                     pass
                 else:
                     warning.append("Publisher "+x+" Affiliation or US Society Missing/Not US Society")#if this works, check all represented composers
         if allresults['Entered Writer Share Total'] > 100.05:
             warning.append("Entered Writer Share Exceeds 100")

         for x in range(1,10):#so this premise works...seems like warning conditions are a little funky though
             x = str(x)
             if (allresults['Client '+x+' Name'] == '#N/A'):             
                  warning.append("Client "+x+" spelling not In Master Meta! Need it set up properly for Territory Info!")
         return warning     

    allresults['warnings'] = allresults.apply(warnings, axis=1)
   
    allresults = allresults[[
        'CWRIndex',
        "warnings"]].to_excel('AdminWarning.xlsx')
    print("Process Finished, Please validate AdminWarning.xlsx") 
    input("Press Enter to exit...")