#!/usr/bin/env python
# coding: utf-8

# Importing relavant libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from get_sink_data import sinkdata
from pandas.core.common import flatten
import re
from scipy.special import gamma as Gamma,gammaincc as GammaInc,factorial
import random 
import matplotlib.patches as mpatches
import copy
import itertools

def adjust_font(lgnd=None, lgnd_handle_size=49, fig=None, ax_fontsize=14, labelfontsize=14,right = True,top = True):
    '''Change the font and handle sizes'''
    #Changing the legend size
    if not (lgnd is None):
        for handle in lgnd.legendHandles:
            handle.set_sizes([lgnd_handle_size])
    #Changing the axis and label text sizes
    if not (fig is None):
        ax_list = fig.axes
        for ax1 in ax_list:
            ax1.tick_params(axis='both', labelsize=ax_fontsize)
            ax1.set_xlabel(ax1.get_xlabel(),fontsize=labelfontsize)
            ax1.set_ylabel(ax1.get_ylabel(),fontsize=labelfontsize)
            ax1.minorticks_on()
            ax1.tick_params(axis='both',which='both', direction='in',top=top,right=right)


#Remove the brown dwarfs for a data
def Remove_Brown_Dwarfs(data,minmass = 0.08):
    '''
Remove the Brown Dwarfs from the initial data file

Inputs
----------
data : list of sinkdata objects
The file that needs to be filtered

Parameters
----------
minmass : float,int,optional
The mass limit for Brown Dwarfs.

Returns
-------
data : list of sinkdata objects
The original file but the brown dwarfs are removed from the x,v,id,formation_time and m parameters

Example
-------
M2e4_C_M_J_2e7 = Remove_Brown_Dwarfs(M2e4_C_M_J_2e7,minmass = 0.08)
            '''
    #Get the lowest snapshot in which there are any stars.
    lowest = 0
    for i in range(len(data)):
        if len(data[i].m)>0:
            break
        lowest = i
    #Change the data to remove the points corresponding to Brown Dwarfs
    for i in range(lowest,len(data)):
        data[i].x = data[i].x[data[i].m>0.08]
        data[i].v = data[i].v[data[i].m>0.08]
        data[i].id = data[i].id[data[i].m>0.08]
        data[i].formation_time = data[i].formation_time[data[i].m>0.08]
        for label_no in range(len(data[i].extra_data_labels)):
            if data[i].extra_data_labels[label_no] == 'ProtoStellarStage' or data[i].extra_data_labels[label_no] == 'ProtoStellarAge':
                data[i].extra_data[label_no] = data[i].extra_data[label_no][data[i].m>0.08]
        data[i].m = data[i].m[data[i].m>0.08]

    return data

#Simple function that can return the closest index or value to another one
def closest(lst,element,param = 'value'):
    '''
Get the closest value to a target element in a given list

Inputs
----------
lst : list,array
The list to find the value from
    
element: int,float
The target element to find the closest value to in the list

Parameters
----------
param : string,optional
The param can be set to value, which returns the closest value in the list or index, which returns the index of the closest value in the list. By default, it returns the value.

Returns
-------
closest_ele : int,float
Either the closest element (if param is value) or the index of the closest element(if param is index)

Example
-------
1) closest([1,2,3,4],3.6,param = 'value')

This returns the value of the element in the list closest to the given element. (4)

2) closest([1,2,3,4],3.6,param = 'index')

This returns the index of the list element closest to the given element. (3)
    '''
    lst = np.asarray(lst) 
    idx = (np.abs(lst - element)).argmin() 
    if param == 'value':
        return lst[idx]
    elif param == 'index':
        return idx

#Removing an item in a nested list
def nested_remove(L, x):
    'Remove an item from a list with multiple layers of nesting'
    if (x in L) :
        L.remove(x)
    else:
        for element in L:
            if type(element) is list:
                nested_remove(element, x)

def findsubsets(s, n):
    'Find all the subsets of a list (s) of length (n).'
    tuplelist = list(itertools.combinations(s, n))
    subset_list = []
    for i in range(len(tuplelist)):
        subset_list.append(list(tuplelist[i]))
    return subset_list

def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True
#This function finds the first snapshot that a star is in
def first_snap_finder(ide,file):
    'Find the first snapshot where an id is present in the file.'
    for i,j in enumerate(file):
        if ide in j.id:
            return i

#This finds the first snapshot that a star is in a certain mass range
def first_snap_mass_finder(ide,file,lower_limit,upper_limit):
        'Find the first snapshot where an id is present in the file in a certain mass range.'
        for i,j in enumerate(file):
            if ide in j.id and lower_limit<=j.m[j.id == ide][0]<=upper_limit:
                return i

def IGamma(k,n):
    '''The Incomplete Gamma Function'''
    return GammaInc(k,n)*Gamma(k)

def sigmabinom(n,k):
    '''The Binomial Error Function'''
    return np.sqrt((k*(n-k))/n**3)

def Psigma(n,k):
    '''Complex Binomial Error Function'''
    variance = (-Gamma(2+n)**2*Gamma(2+k)**2)/(Gamma(3+n)**2*Gamma(1+k)**2)+(Gamma(3+k)*Gamma(2+n))/(Gamma(1+k)*Gamma(4+n))
    return np.sqrt(variance)

def Lsigma(n,k):
    '''Multiplicity Frequency Error Function'''
    variance = -((Gamma(2+k)-IGamma(2+k,3*n))**2/(n**2*(Gamma(1+k)-IGamma(1+k,3*n))**2))+((Gamma(3+k)-IGamma(3+k,3*n))/(n**2*(Gamma(1+k)-IGamma(1+k,3*n))))
    return np.sqrt(variance)

def load_files(filenames,brown_dwarfs = False):
    '''
Initialize data from provided filenames.

Inputs
----------
filenames : string (single file) or list of strings (multiple files)
Input the file name(s) that you would like to initialize data from.

Parameters
----------
brown_dwarfs : bool,optional
Removes all Brown Dwarfs from the file if False.

Returns
-------
Files_List : array
The files that were requested in an array.

Example
-------
files = load_files([filename_1,filename_2,...],brown_dwarfs = False)
            '''
    if isinstance(filenames,str):
        filenames = [filenames] #If the input is a string
    Files_List = []
    for i in tqdm(range(len(filenames))):
        #Load the file for all files
        pickle_file = open(filenames[i]+str('.pickle'),'rb')
        data_file = pickle.load(pickle_file)
        pickle_file.close()
        if brown_dwarfs == False:
            filtered_data_file = Remove_Brown_Dwarfs(data_file)
        else:
            filtered_data_file = data_file
        Files_List.append(filtered_data_file) 

    return np.array(Files_List)

## This function calculates Binding Energy
def Binding_Energy(m1,m2,x1,x2,v1,v2,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    'Calculate the Binding Energy (in J) from given masses,positions and velocities. If using a box file, provide the lengths to modify it.'
    mu = (m1*m2)/(m1+m2)
    KE = 1.9891e30 * 0.5 * ((v1[0]-v2[0])**2+(v1[1]-v2[1])**2 +(v1[2]-v2[2])**2) * mu
    dx = x1[0]-x2[0];dy = x1[1]-x2[1];dz = x1[2]-x2[2]
    #If the edge is periodic, we replace the long distance with the short distance in 1D.
    if periodic_edge is True:
        if dx > Lx/2:
            dx = Lx - dx
        if dy > Ly/2:
            dy = Ly - dy
        if dz > Lz/2:
            dz = Lz - dz
    PE = ((1.9891e30)**2 *6.67e-11*(m1+m2)*mu)/((3.08567758e16)*np.sqrt(dx**2+dy**2+dz**2))
    E = KE - PE
    return E

#This function is able to calculate the binding energy matrix of a set of nodes
def Binding_Energy_Matrix(m,x,v,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    'Calculate the Binding Energy Matrix (in J) from a list of masses,positions and velocities. If using Box run, provide the edge lengths.'
    Binding_energy_matrix = np.zeros((len(m),len(m)))
    for i in range(len(m)):
        for j in range(len(m)):
            if Binding_energy_matrix[i,j] == 0: 
                E = 0
                if i == j:
                    E = float('inf')
                else:
                    E = Binding_Energy(m[i],m[j],x[i],x[j],v[i],v[j],periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz) 
                Binding_energy_matrix[i][j] = E                
                Binding_energy_matrix[j][i] = E
    return Binding_energy_matrix

## This is done to speed up runtime and help remove outliers
# Since the pairs normally have a seperation of just 0.1 pc, we can just divide them into large chuncks of ~2-3 pc instead

#The min and max function is used on the coordinates to get the max x,y and z depending on the dimension
def min_and_max(x,dimension = 'x'):
    '''
Find the minimum and maximum value along a dimension 
Inputs
----------
x : list,array
The array of positions.

Parameters
----------
dimension : 'x', 'y' or 'z', string
The dimension along which to find the minimum or maxium. 

Returns
-------
min_x : float,int
The minimum position along the given dimension.
    
max_x : float,int
The maximum position along the given dimension.

Example
-------
min_x,max_x = min_and_max(data[snapshot].x,dimension = 'x')

    '''
    if dimension == 'x':
        n = 0
    elif dimension == 'y':
        n = 1
    elif dimension == 'z':
        n = 2
    min_x = np.inf
    max_x = -np.inf
    for i in x:
        if i[n] < min_x:
            min_x = i[n]
        if i[n] > max_x:
            max_x = i[n]
    return min_x,max_x

def Splitting_Data(file,snapshot,seperation_param,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    '''
Splitting the given files data into bins of a defined length. This would put the masses,positions, velocities and ids into different regions corresponding to
the split regions.
Inputs
----------
file : list of sinkdata
The data file to which the splitting will be applied
    
snapshot: int
The snapshot to which to apply splitting
    
seperation_param : float,int
The size of the box you want to split (in pc). 
Returns
-------
m,x,v,ids : lists
The masses,positions,velocities and ids put into different lists corresponding to different bins.

Example
-------
1) m,x,v,ids = Splitting_Data(file = M2e4_C_M_J_2e7_BOX,snapshot = -1,seperation_param = 2,periodic_edge = True,Lx = 1,Ly = 1,Lz = 1)

This is for a box file with side lengths 1 and the splitting happening every 2 pc.

2) m,x,v,ids = Splitting_Data(file = M2e4_C_M_J_2e7,snapshot = -1,periodic_edge = False)

This is for a non box file and the splitting happening every 2 pc.

    '''
    # Using the max_and_min function
    min_x,max_x = min_and_max(file[snapshot].x,'x')
    min_y,max_y = min_and_max(file[snapshot].x,'y')
    min_z,max_z = min_and_max(file[snapshot].x,'z')
    # Defining a seperation parameter, the smaller the faster the program will run but the less accurate it will be
    #Doing the clustering using the digitize function for x,y and z seperately. Since the bins aren't zero indexed, they are 
    # all subtracted by one
    if periodic_edge == False:
        bins_x = np.arange(min_x,max_x,seperation_param)
        clusters_x = np.digitize(file[snapshot].x,bins_x)
        for i in range(len(clusters_x)):
            clusters_x[i] = clusters_x[i]-1

        bins_y = np.arange(min_y,max_y,seperation_param)
        clusters_y = np.digitize(file[snapshot].x,bins_y)
        for i in range(len(clusters_y)):
            clusters_y[i] = clusters_y[i]-1

        bins_z = np.arange(min_z,max_z,seperation_param)
        clusters_z = np.digitize(file[snapshot].x,bins_z)
        for i in range(len(clusters_z)):
            clusters_z[i] = clusters_z[i]-1
    else:
        bins_x = np.arange(seperation_param,Lx-seperation_param,seperation_param)
        bins_x = np.insert(bins_x,0,0)
        bins_x = np.append(bins_x,Lx+0.1)
        clusters_x = np.digitize(file[snapshot].x,bins_x)
        for i in range(len(clusters_x)):
            clusters_x[i] = clusters_x[i]-1

        bins_y = np.arange(seperation_param,Ly-seperation_param,seperation_param)
        bins_y = np.insert(bins_y,0,0)
        bins_y = np.append(bins_y,Ly+0.1)
        clusters_y = np.digitize(file[snapshot].x,bins_y)
        for i in range(len(clusters_y)):
            clusters_y[i] = clusters_y[i]-1

        bins_z = np.arange(seperation_param,Lz-seperation_param,seperation_param)
        bins_z = np.insert(bins_z,0,0)
        bins_z = np.append(bins_z,Lz+0.1)
        clusters_z = np.digitize(file[snapshot].x,bins_z)
        for i in range(len(clusters_z)):
            clusters_z[i] = clusters_z[i]-1
    # The x,m,v and ids are first initialized to empty list of lists. Then, they get the required indexes from the earlier
    #function and use that to place the objects in the correct indices
    x = np.zeros((len(bins_x),len(bins_y),len(bins_z))).tolist()
    m = np.zeros((len(bins_x),len(bins_y),len(bins_z))).tolist()
    v = np.zeros((len(bins_x),len(bins_y),len(bins_z))).tolist()
    ids = np.zeros((len(bins_x),len(bins_y),len(bins_z))).tolist()
    for i in range(len(m)):
        for j in range(len(m[0])):
            for k in range(len(m[0][0])):
                x[i][j][k] = []
                m[i][j][k] = []
                v[i][j][k] = []
                ids[i][j][k] = []
    for i in range(len(file[snapshot].x)):
        iindex = clusters_x[i][0]
        jindex = clusters_y[i][1]
        zindex = clusters_z[i][2]
        x[iindex][jindex][zindex].append(file[snapshot].x[i])
        m[iindex][jindex][zindex].append(file[snapshot].m[i])
        v[iindex][jindex][zindex].append(file[snapshot].v[i])
        ids[iindex][jindex][zindex].append(file[snapshot].id[i])
    if periodic_edge == True:
        for i in range(len(bins_x)):
            for j in range(len(bins_y)):
                for k in range(len(bins_z)):
                    if i == 0 or i == len(bins_x)-1 or j == 0 or j == len(bins_y)-1 or k == 0 or k == len(bins_z)-1:
                        if not (i == 0 and j == 0 and k==0):
                            m[0][0][0].extend(m[i][j][k])
                            m[i][j][k] = []
                            x[0][0][0].extend(x[i][j][k])
                            x[i][j][k] = []
                            v[0][0][0].extend(v[i][j][k])
                            v[i][j][k] = []
                            ids[0][0][0].extend(ids[i][j][k])
                            ids[i][j][k] = []
                            
            
    return m,x,v,ids

## Most crucial function of the program (Program's runtime comes mainly from here)
def remove_and_replace(matrix,m,x,v,ids,periodic_edge = False,Lx = None,Ly = None,Lz = None): 
    '''
Find the minimum value in the Binding Energy Matrix and changes the masses, positions, velocities and ids of that node.

Inputs
----------
matrix : list, nested list
The binding energy matrix to be analyzed.
    
m: list, array
The masses to be changed.
    
x : list, array
The positions to be changed.
    
v : list, array
The velocities to be changed.
    
ids : list, array
The ids to be changed.
    
periodic_edge : bool,optional
If you are using the box simulation with periodic edges.

Lx: int,float,optional
The length of periodic box in the x direction.

Ly: int,float,optional
The length of periodic box in the y direction.

Lz: int,float,optional
The length of periodic box in the z direction.

Returns
-------

new_matrix : list,nested list
    The new matrix with the rows and columns corresponding to the minimum element deleted.
    
indexes : list
    The new ids with the previous min element's ids replaced with one list with both the ids.

new_masses: list
    The new masses with the previous min element's masses replaced with one mass that is the sum of the ids.

new_x : list
    The new ids with the previous min element's positions replaced with the position of the center of mass.

new_v: list
    The new ids with the previous min element's velocities replaced with the velocity of the center of mass.

Example
-------
1) new_matrix,indexes,new_masses,new_x,new_v = remove_and_replace(matrix,m,x,v,ids,periodic_edge = True,Lx = 1,Ly = 1,Lz = 1)

This is for a pre determined matrix,m,x,v and ids for a box file with side lengths 1.

2) new_matrix,indexes,new_masses,new_x,new_v = remove_and_replace(matrix,m,x,v,ids,periodic_edge = False)

This is for a pre determined matrix,m,x,v and ids for a non-box file.

    '''
#Optimized version of the remove_and_replace. This first finds the minimum element using a function(thus not looping through
#the entire matrix) and then checks for two things, first, if its greater than 0 (unbound), then it stops the while loop or if the system it
#is working on fits the criteria of 4 or less objects, it stops the while loop
    indexes = list(ids)
    most_bound_element = 0
    most_bound_element_indices = [0,0]
    comp_matrix = matrix
    flag = 0
    while flag == 0:
        most_bound_element = comp_matrix.min()
        if most_bound_element > 0:
            flag = 1
        most_bound_element_indices = list(np.unravel_index(comp_matrix.argmin(), comp_matrix.shape))
        most_bound_element_indices.sort()
        if (len(list(flatten(list([indexes[most_bound_element_indices[0]],indexes[most_bound_element_indices[1]]])))))>4:
            comp_matrix[most_bound_element_indices[0]][most_bound_element_indices[1]] = np.inf
            comp_matrix[most_bound_element_indices[1]][most_bound_element_indices[0]] = np.inf
        else:
            flag = 1
#If the most bound element is infinity, that means that we couldn't find any other
#objects that fit the 4 objects criteria because we've replaced them all, so instead we return the input back
    if most_bound_element > 0:
        return matrix,ids,m,x,v
    else:
        small_i = most_bound_element_indices[0]
        big_i = most_bound_element_indices[1]
    #Creates the new indexes where the ids are joined
        indexes[small_i] = list([ids[small_i],ids[big_i]])
        del indexes[big_i]
    # Defining the new object's mass as the reduced mass of the initial ones, the coordinates as the CoM of the initial ones
    # and the velocity as the CoM velocity of the initial ones & also defining new mass,x and v from the CoM and total mass
    # and also getting the new x,v, and m from the total mass and CoM of x and v
        new_object_x = [0,0,0]
        new_object_v = [0,0,0]
        new_object_mass = (m[small_i]+m[big_i])
        
        new_x = list(x)
        new_v = list(v)
        new_masses = list(m)
        
        for i in range(0,3):
            new_object_x[i] = (((new_masses[big_i]*new_x[big_i][i])+(new_masses[small_i]*new_x[small_i][i]))/new_object_mass)
            new_object_v[i] = (((new_masses[big_i]*new_v[big_i][i])+(new_masses[small_i]*new_v[small_i][i]))/new_object_mass)
        
        new_x[small_i] = new_object_x
        new_x = np.delete(new_x,big_i,axis = 0)
        new_v[small_i] = new_object_v
        new_v = np.delete(new_v,big_i,axis = 0)
        new_masses[small_i] = new_object_mass
        new_masses = np.delete(new_masses,big_i)
    # Deleting one column and row corresponding to the smaller index, which makes the bigger index decrease by 1
        matrix = np.delete(matrix,most_bound_element_indices[1],0)
        matrix = np.delete(matrix,most_bound_element_indices[1],1)
        replace_indice = most_bound_element_indices[0]
    # Creating a new matrix and replacing the row and column with the replace index with the binding energies of the new object
    # and the other objects
        new_matrix = matrix
        Binding_Energy_row = []
        for i in range(len(new_masses)):
            if i == replace_indice:
                Binding_Energy_row.append(np.float('inf'))
            else: 
                Energy_of_these_objects = Binding_Energy(new_masses[i],new_object_mass,new_x[i],new_object_x,new_v[i],new_object_v,periodic_edge=periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz)
                Binding_Energy_row.append(Energy_of_these_objects)
        new_matrix[replace_indice] = Binding_Energy_row
        for i in range(len(new_matrix)):
            new_matrix[i][replace_indice] = Binding_Energy_row[i]
    # Now the new matrix has the information we want. Thus, we'll want to send back the new masses, coordinates, velocities,
    #the new matrix and the indices of the removed objects
        return new_matrix,indexes,new_masses,new_x,new_v

# Since the previous function had the constraints already, we can make the while condition stop when there is no additional 
# clustering happening 
def constrained_remove_and_replace(binding_energy_matrix,ids,m,x,v,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    '''
Perform the remove and replace until it is no longer possible (because there are no more bound systems of less than 4 stars).

Inputs
----------
binding_energy_matrix : list, nested list
The binding energy matrix to be analyzed.

ids : list, array
The ids to be changed.
    
m: list, array
The masses to be changed.
    
x : list, array
The positions to be changed.
    
v : list, array
The velocities to be changed.

periodic_edge : bool,optional
If you are using the box simulation with periodic edges.

Lx: int,float,optional
The length of periodic box in the x direction.

Ly: int,float,optional
The length of periodic box in the y direction.

Lz: int,float,optional
The length of periodic box in the z direction.

Returns
-------
ids: nested list
The new ids containing the ids arranged by system.

Example
-------
1) ids = constrained_remove_and_replace(matrix,ids,m,x,v,periodic_edge = True,Lx = 1,Ly = 1,Lz = 1)

This is for a pre determined matrix,m,x,v and ids for a box file with side lengths 1.

2) ids = constrained_remove_and_replace(matrix,ids,m,x,v,periodic_edge = False)

This is for a pre determined matrix,m,x,v and ids for a non-box file.
    '''
    previous_ids = []
    # Wait till no more objects are clustered
    while len(previous_ids) != len(ids):
        previous_ids = list(ids)
        binding_energy_matrix, ids, m, x, v = remove_and_replace(binding_energy_matrix,m,x,v,ids,periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz)
    return ids

#This part of the program makes use of all the previous function definitions and leads to the result
def clustering_algorithm(file,snapshot_number,seperation_param = 2,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    '''
The main algorithm that can perform system assignment after splitting the data into boxes.
Inputs
----------
file : list of sinkdata.
The input data that will be grouped into systems.

snapshot_number : int
The snapshot number to use.

periodic_edge : bool,optional
If you are using the box simulation with periodic edges.

Lx: int,float,optional
The length of periodic box in the x direction.

Ly: int,float,optional
The length of periodic box in the y direction.

Lz: int,float,optional
The length of periodic box in the z direction.

Parameters
----------
seperation_param : int, float
The seperation of the boxes. By default, this is 2 pc and that is the separation of the pickle files. 

Returns
-------
Result: list
The new ids containing the ids arranged by system.

Example
-------
1) Result = clustering_algorithm(M2e4_C_M_J_2e7,-1,seperation_param = 2,periodic_edge = True,Lx = 1,Ly = 1,Lz = 1)

This is for a box file with side lengths 1 and split every 2 pc.

2) Result = clustering_algorithm(M2e4_C_M_J_2e7,-1,seperation_param = 2)

This is for a non-box file that is split every 2 pc.
    '''
    #If the file has one or less stars, we just return the ids as is
    if len(file[snapshot_number].m) <=1:
        return file[snapshot_number].id
    #Otherwise, we perform out algorithm on the nodes.
    else:
        m,x,v,idess = Splitting_Data(file,snapshot_number,seperation_param,periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz)

        Result = []
        for i in tqdm(range(len(m)),position = 0,desc = 'Main Loop',leave= True ):
            for j in range(len(m[i])):
                for k in range(len(m[i][j])):
                    Binding = Binding_Energy_Matrix(m[i][j][k],x[i][j][k],v[i][j][k],periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz)
                    Result.append(constrained_remove_and_replace(Binding,idess[i][j][k],m[i][j][k],x[i][j][k],v[i][j][k],periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz))


        return Result

def clustering_algorithm_no_split(file,snapshot_number,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    '''
The main algorithm that can perform system assignment without splitting.
Inputs
----------
file : list of sinkdata.
The input data that will be grouped into systems.

snapshot_number : int
The snapshot number to use.

periodic_edge : bool,optional
If you are using the box simulation with periodic edges.

Lx: int,float,optional
The length of periodic box in the x direction.

Ly: int,float,optional
The length of periodic box in the y direction.

Lz: int,float,optional
The length of periodic box in the z direction.

Returns
-------
Result: list
The new ids containing the ids arranged by system.

Example
-------
1) Result = clustering_algorithm(M2e4_C_M_J_2e7,-1,seperation_param = 2,periodic_edge = True,Lx = 1,Ly = 1,Lz = 1)

This is for a box file with side lengths 1.

2) Result = clustering_algorithm(M2e4_C_M_J_2e7,-1,seperation_param = 2)

This is for a non-box file.
    '''
    if len(file[snapshot_number].m) <=1:
        return file[snapshot_number].id
    else:
        m = file[snapshot_number].m
        x = file[snapshot_number].x
        v = file[snapshot_number].v
        idess = file[snapshot_number].id

        Result = []
        Binding = Binding_Energy_Matrix(m,x,v,periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz)
        Result.append(constrained_remove_and_replace(Binding,idess,m,x,v,periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz))


        return Result

# Defining a class that contains the structured & non-structured ids, the masses, the snapshot number, the coordinates, the 
# velocities, the primary mass, the secondary mass, their ratio and the total mass of the system
class star_system:
    def  __init__(self,ids,n,data):
        self.structured_ids = ids #Saving the structured ids
        if isinstance(ids,list): #Flattening the ids
            self.ids = list(flatten(list(ids)))
            self.no = len(self.ids)
        else:
            self.ids = [ids]
            self.no = 1
        #ai = []
        #u = np.in1d(data[n].id,self.ids)
        m = []
        x = []
        v = []
        for i in self.ids:
            m.append(data[n].m[data[n].id == i][0])# Getting the masses
            x.append(data[n].x[data[n].id == i][0])# Getting the positions 
            v.append(data[n].v[data[n].id == i][0])# Getting the velocities
        self.m = np.array(m)
        self.x = np.array(x)
        self.v = np.array(v)
       # for i in range(len(u)):
        #    if u[i] == True:
         #       ai.append(i)
        #self.m = data[n].m[ai]
        #self.x = data[n].x[ai]
        #self.v = data[n].v[ai]
        self.snapshot_num = n #The snapshot number of the system
        self.tot_m = sum(self.m) #The total mass of the system
        primary_mass = max(self.m) #The primary (most massive) star in the system
        self.primary = primary_mass 
        self.primary_id = self.ids[np.argmax(self.m)] #The primary star's id
        secondary_mass = 0
        for i in range(len(self.m)):
            if self.m[i] < primary_mass and self.m[i]> secondary_mass:
                secondary_mass = self.m[i]
                sec_ind = i
        self.secondary = secondary_mass #The mass of the second most massive star
        self.mass_ratio = secondary_mass/primary_mass #The companion mass ratio (secondary/primary)
        
        #Now defining the semi-major axis(Check for circular orbit and only use targeted primary masses again to plot dist.)
        #Do not use this except for binaries
        #Note: The semi major axis is in m
        if self.no>1 and self.secondary>0:
            vel_prim = self.v[np.argmax(self.m)]
            x_prim = self.x[np.argmax(self.m)]
            vel_sec = self.v[sec_ind]
            x_sec = self.x[sec_ind]
            binding_energy = Binding_Energy(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
            mu = 1.9891e30*((primary_mass*secondary_mass)/(primary_mass+secondary_mass))
            eps = binding_energy/mu
            mu2 = 1.9891e30*6.67e-11*(primary_mass+secondary_mass)
            self.smaxis = - (mu2/(2*eps))
        else:
            self.smaxis = 0

# Main Function of the program
def system_creation(file,snapshot_num,Master_File,read_in_result = False,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    '''
The main function that does the system assignment(with splitting) and makes them star system objects.
Inputs
----------
file : list of sinkdata.
The input data that will be grouped into systems.

snapshot_number : int
The snapshot number to use.
    
Master_File : list of star system lists:
The file containing already assigned systems to use if you don't want to apply algorithm again.

periodic_edge : bool,optional
If you are using the box simulation with periodic edges.

Lx: int,float,optional
The length of periodic box in the x direction.

Ly: int,float,optional
The length of periodic box in the y direction.

Lz: int,float,optional
The length of periodic box in the z direction.

Parameters
----------
read_in_result : bool
Whether to read in the result from pickle files or do system assignment.

Returns
-------
systems: list of star system objects
The list of star system objects.

Example
-------
1) systems = system_creation(M2e4_C_M_2e7,-1,M2e4_C_M_2e7_systems,read_in_result = True,periodic_edge = False)

This is the example of creating the systems for a non-boxed file with the systems already made (in the form of Master_File)

2) systems = system_creation(M2e4_C_M_2e7,-1,M2e4_C_M_2e7_systems,read_in_result = True,periodic_edge = True,Lx = L,Ly = L,Lz = L)

This is the example of creating the systems for a boxed file with the systems already made (in the form of Master_File)

3) systems = system_creation(M2e4_C_M_2e7,-1,'placeholder_text',read_in_result = False,periodic_edge = False)

This is the example of creating the systems for a non-boxed file where the systems aren't already made.

    '''
    systems = []
    #If you have the file, you can read it in from a premade Master_File, otherwise, you perform the algorithm
    if read_in_result == True:
        return Master_File[snapshot_num]
    else:
        Result = clustering_algorithm(file,snapshot_num,periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz)
   #Turn the id pairs into star system objects
    for i in tqdm(Result,desc = 'System Creation',position = 0,leave = True):
        if isinstance(i,list):
            for j in i:
                systems.append(star_system(j,snapshot_num,file))
        else:
            systems.append(star_system(i,snapshot_num,file))
    return systems    

def system_creation_no_splitting(data,snapshot_num,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    '''
The main function that does the system assignment(without splitting) and makes them star system objects.
Inputs
----------
data : list of sinkdata.
The input data that will be grouped into systems.

snapshot_number : int
The snapshot number to use.

periodic_edge : bool,optional
If you are using the box simulation with periodic edges.

Lx: int,float,optional
The length of periodic box in the x direction.

Ly: int,float,optional
The length of periodic box in the y direction.

Lz: int,float,optional
The length of periodic box in the z direction.

Returns
-------
systems: list of star system objects
The list of star system objects.

Example
-------
1) systems = system_creation_no_splitting(M2e4_C_M_2e7,-1,periodic_edge = False)

This is the example of creating the systems for a non-boxed file without splitting.

2) systems = system_creation_no_splitting(M2e4_C_M_2e7,-1,periodic_edge = True,Lx = L,Ly = L,Lz = L)

This is the example of creating the systems for a boxed file of side length L without splitting.


    '''

    systems = []
    Result = clustering_algorithm_no_split(data,snapshot_num,periodic_edge = periodic_edge,Lx = Lx,Ly = Ly,Lz = Lz)
   
    for i in tqdm(Result,desc = 'System Creation',position = 0,leave = True):
        if isinstance(i,list):
            for j in i:
                systems.append(star_system(j,snapshot_num,data))
        else:
            systems.append(star_system(i,snapshot_num,data))
    return systems   

#This is an SFE finder, gives you the snapshot for an SFE by finding the closest value to the target SFE. It also gives you
#that SFE so that you know how close it is
def SFE_snapshot(file,SFE_param = 0.04):
    '''Figure out the closest snapshot in a file to a certain SFE value'''
    pickle_file = open(file,'rb')
    data = pickle.load(pickle_file) 
    pickle_file.close()
    initmass = np.float(re.search('M\de\d', file).group(0).replace('M',''))
    mass_sum = []
    for j in data:
        mass_sum.append(sum(j.m))
    SFE = []
    for j,l in enumerate(mass_sum):
        SFE.append(l/initmass)
    snap = closest(SFE,SFE_param,param='index')
    SFE_at = closest(SFE,SFE_param,param='value')
    print('SFE in the closest snapshot is '+str(SFE_at))
    return snap

#Finds the first snapshot with more than one star over a mass (i.e the first instance of creation of that mass)
def Mass_Creation_Finder(file,min_mass = 1):
    '''Find the first snapshot in a file which there is at least one star over a certain mass. '''
    snap = 0
    for i in range(len(file)):
        if len(file[i].m[file[i].m>min_mass]) != 0 and len(file[i].m>0):
            snap = i
            break
    return snap

def system_initialization(file,file_name,read_in_result = True,full_assignment = False,snapshot_num = -1,periodic_edge = False,Lx = None,Ly = None,Lz = None):
    '''
This function initializes the systems for a given file.
Inputs
----------
file : list of sinkdata.
The input data that will be grouped into systems.
    
file_name: string
The name of the file which will be matched to the system pickle file.

Parameters
----------
read_in_result : bool,optional
Whether to read in the result from pickle files or do system assignment.

full_assignment: bool,optional
Whether to perform system assignment on all snapshots.
    
snapshot_num: int,optional
The snapshot to perform assignment if you only want to do it for one snap (i.e full_assignment = False).

periodic_edge : bool,optional
If you are using the box simulation with periodic edges.

Lx: int,float,optional
The length of periodic box in the x direction.

Ly: int,float,optional
The length of periodic box in the y direction.

Lz: int,float,optional
The length of periodic box in the z direction.

Returns
-------
systems: list of star system objects,list of list of star system objects
The list of all star systems from each snap or for one snap in the file. Note: The best variable name to save this is something like 'filename_systems'.

Example
-------
1) M2e4_C_M_2e7_systems = system_initialization(M2e4_C_M_2e7,'M2e4_C_M_2e7',read_in_result = True)

This is the example of creating the systems for a non-boxed file where the systems are already made.

2) M2e4_C_M_2e7_systems = system_initialization(M2e4_C_M_2e7,'M2e4_C_M_2e7',read_in_result = True,periodic_edge = True, Lx = L,Ly = L,Lz = L)

This is the example of creating the systems for a boxed file of length L where the systems are already made.

3) M2e4_C_M_2e7_systems = system_initialization(M2e4_C_M_2e7,'M2e4_C_M_2e7',read_in_result = False,snapshot_num = -1)

This is the example of creating the systems for a single snapshot for a non-boxed file where the systems are not already made.

4) M2e4_C_M_2e7_systems = system_initialization(M2e4_C_M_2e7,'M2e4_C_M_2e7',read_in_result = False,full_assignment = True)

This is the example of creating all the systems for a non-boxed file where the systems are not already made.



    '''
    if read_in_result == True:
        infile = open(file_name+str('_Systems.pickle'),'rb')
        Master_File = pickle.load(infile)
        infile.close()
        return Master_File #Simply opening the pickle file
    elif read_in_result == False:
        if full_assignment == True:
            Result_List = []
            for i in tqdm(range(len(file)),desc = 'Full Assignment',position = 0):
                print('Snapshot No: '+str(i)+'/'+str(len(file)-1))
                Result_List.append(system_creation(file,i,Master_File = file,read_in_result = False,periodic_edge = False,Lx = None,Ly = None,Lz = None))
            return Result_List #Returning the list of assigned systems
        else:#Returning just one snapshot
            return system_creation(file,snapshot_num,Master_File = file,read_in_result = False,periodic_edge = False,Lx = None,Ly = None,Lz = None)

#Convert the simulation time to Myrs
time_to_Myr = 978.461942384

#Convert AU to m (divide by this if you have a result in m)
m_to_AU = 149597870700.0

#Convert pc to AU
pc_to_AU = 206264.806

#This is a filter for minimum q for one snapshot
def q_filter_one_snap(systems,min_q = 0.1):
    #Change Master_File to file snapshot
    '''The q filter as applied to one snapshot'''
    Filtered_Master_File = copy.deepcopy(systems) #Creating a new copy of the master file
    for i,j in enumerate(Filtered_Master_File):
        if j.no>1:
            for k in j.m:
                if k/j.primary < min_q:
                    if j.no == 4:
                        state = 0 #We need to see if its a [[1,2],[3,4]] or [1,[2,[3,4]]] system
                        for idd in j.structured_ids:
                            if isinstance(idd,list):
                                state += len(idd)
                    remove_id = np.array(j.ids)[j.m == k] #The id that we have to remove
                    j.ids.remove(remove_id)
                    j.x = j.x[j.m != k]
                    j.v = j.v[j.m != k]
                    j.no -= 1
                    j.m = j.m[j.m != k]
                    j.tot_m = sum(j.m)
                    if j.no == 1:
                        j.mass_ratio = 0
                        j.secondary = 0 #Remove the secondary if the remaining star is solitary
                        j.structured_ids = [j.ids]
                    if j.no == 2:
                        j.structured_ids = j.ids #The secondary isn't going to be removed if there's 2 stars remaining
                    if j.no == 3:
                        removed_list = copy.deepcopy(j.structured_ids) 
                        checker = remove_id[0] #The remove ID is in an array so we make it single
                        checker = float(checker) #It is an np float so we make it a float
                        nested_remove(removed_list,checker)
                        if state == 4:
                            for index,value in enumerate(removed_list):
                                if isinstance(value,list):
                                    if len(value) == 1:
                                        removed_list[index] = value[0]
                        elif state == 2:
                            if len(removed_list) == 1:
                                removed_list = removed_list[0]
                            if len(removed_list) == 2:
                                for index,value in enumerate(removed_list):
                                    if isinstance(value,list) and len(value) == 1:
                                        removed_list[index] = value[0]
                                    elif isinstance(value,list) and len(value) == 2:
                                        removed_list[index] = list(flatten(value)) 
                        j.structured_ids = removed_list
                    Filtered_Master_File[i] = j
    return Filtered_Master_File

#This function applies the q filter to all snapshots
def q_filter(Master_File):
    '''Applying the q filter to an entire file'''
    New_Master_File = []
    for i in tqdm(Master_File,position = 0,desc = 'Full File q filter loop'):
        appending = q_filter_one_snap(i)
        New_Master_File.append(appending)
    return New_Master_File

def simple_filter_one_system(system,Master_File,comparison_snapshot = -2):
    'Removing every companion that didnt exist for the last 10 Myrs.'
    was_primary_there = False
    for previous_sys in Master_File[comparison_snapshot]:
        if system.primary_id == previous_sys.primary_id:
            previous_target_system = previous_sys
            was_primary_there = True
    if was_primary_there == False: #Just checking if there are snapshots where the primary wasn't formed before
        system.no = 1
        system.ids = [system.primary_id]
        system.secondary = 0
        system.x = system.x[system.m == system.primary]
        system.v = system.v[system.m == system.primary]
        system.m = np.array([system.primary])
        system.mass_ratio = 0
        system.tot_m = system.primary
        system.structured_ids = [system.primary_id]
        return system
    og_system = copy.copy(system)
    for ides in og_system.ids: #Checking all the ids in the snap
        if ides not in previous_target_system.ids and ides != og_system.primary_id: #If any of the companions arent there
            if system.no == 4:
                state = 0 #We need to see if its a [[1,2],[3,4]] or [1,[2,[3,4]]] system
                for idd in og_system.structured_ids:
                    if isinstance(idd,list):
                        state += len(idd)
            remove_mass = system.m[np.array(system.ids) == ides]
            system.ids.remove(ides)
            system.x = system.x[system.m != remove_mass]
            system.v = system.v[system.m != remove_mass]
            system.no -= 1
            system.m = system.m[system.m != remove_mass]
            system.tot_m = sum(system.m)
            if system.no == 1:
                system.mass_ratio = 0
                system.secondary = 0 #Remove the secondary if the remaining star is solitary
                system.structured_ids = [system.primary_id]
            if system.no == 2:
                system.structured_ids = system.ids
                secondary = 0
                for j in system.m:
                    if j < system.primary and j > secondary:
                        secondary = j
                system.secondary = secondary
                system.mass_ratio = secondary/system.primary
            if system.no == 3:
                removed_list = copy.deepcopy(system.structured_ids) 
                nested_remove(removed_list,float(ides))
                if state == 4:
                    for index,value in enumerate(removed_list):
                        if isinstance(value,list):
                            if len(value) == 1:
                                removed_list[index] = value[0]
                elif state == 2:
                    if len(removed_list) == 1:
                        removed_list = removed_list[0]
                    if len(removed_list) == 2:
                        for index,value in enumerate(removed_list):
                            if isinstance(value,list) and len(value) == 1:
                                removed_list[index] = value[0]
                            elif isinstance(value,list) and len(value) == 2:
                                removed_list[index] = list(flatten(value))
                system.structured_ids = removed_list
                secondary = 0
                for j in system.m:
                    if j < system.primary and j > secondary:
                        secondary = j
                system.secondary = secondary
                system.mass_ratio = secondary/system.primary
                #Add secondary
    return system

def full_simple_filter(Master_File,file,selected_snap = -1,long_ago = 0.5):
    if file[selected_snap].t*time_to_Myr<long_ago:
        #We cant look at a snapshot before 0.5 Myr 
        print('The selected snapshot is too early to use')
        return np.nan
    snap_1 = selected_snap-1
    snap_2 = selected_snap-2
    times = []
    Filtered_Master_File = copy.deepcopy(Master_File)
    for i in tqdm(file,desc = 'Times'):
        times.append(i.t*time_to_Myr)
    snap_3 = closest(times,file[selected_snap].t*time_to_Myr - long_ago,param = 'index')
    
    
    for system_no,system in enumerate(tqdm(Filtered_Master_File[selected_snap],desc = 'Simple Filter Loop',position = 0)):
        result_1 = simple_filter_one_system(system,Filtered_Master_File,comparison_snapshot=snap_1)
        result_2 = simple_filter_one_system(result_1,Filtered_Master_File,comparison_snapshot=snap_2)
        result_3 = simple_filter_one_system(result_2,Filtered_Master_File,comparison_snapshot=snap_3)
        Filtered_Master_File[selected_snap][system_no] = result_3
    return Filtered_Master_File[selected_snap]

def default_GMC_R(initmass = 2e4):
    '''Make the default R 10 pc for a GMC of 2e4 '''
    return 10

def file_properties(filename,param = 'm'):
    '''Get the initial properties of the cloud from the file name.'''
    #Lets get the initial gas mass for each, which we can only get from the name
    f = filename
    initmass=np.float(re.search('M\de\d', f).group(0).replace('M',''))
    if re.search('R\d', f) is None:
        R=default_GMC_R(initmass)
    else:
        R=np.float(re.search('R\d\d*', f).group(0).replace('R',''))
    if re.search('alpha\d', f) is None:
        alpha=2.0
    else:
        alpha=np.float(re.search('alpha\d*', f).group(0).replace('alpha',''))
    if 'Res' in f:
        npar=np.float(re.search('Res\d*', f).group(0).replace('Res',''))**3
    else:
        npar=np.float(re.search('_\de\d', f).group(0).replace('_',''))
    if param == 'm':
        return initmass
    elif param == 'r':
        return R
    elif param == 'alpha':
        return alpha
    elif param == 'res':
        return npar
def t_ff(mass,R):
    '''Calculate the freefall time'''
    G_code=4325.69
    tff = np.sqrt(3.0*np.pi/( 32*G_code*( mass/(4.0*np.pi/3.0*(R**3)) ) ) )
    return tff

def new_stars_count(file,plot = True,time = True,all_stars = False,lower_limit = 0,upper_limit = 10000):
    '''
The count of new stars or all stars of a certain mass range formed at different snapshots.
Inputs
----------
file : list of sinkdata.
The input data that will be grouped into systems.

Parameters
----------
plot : bool,optional
Whether to plot the number of stars.

time: bool,optional
Whether to have snapshot number or time as the x axis.
    
all_stars: bool,optional
Whether to calculate all stars at a snapshot or just the new stars.

lower_limit: int,float,optional
The lower limit of the mass range

upper_limit: int,float,optional
The upper limit of the mass range

Returns
-------
no_of_stars: list
Either the list of number of new stars or the number of total stars at each snapshot.

Example
-------
1) new_stars_count(M2e4_C_M_2e7,time = True)
Plotting the new stars count over time.

2) new_stars_count(M2e4_C_M_2e7,all_stars = True)
Plotting the total stars over time.

3) new_stars_count(M2e4_C_M_2e7,time = True,lower_limit = 0,upper_limit = 1)
Plotting the new stars count between 0 and 1 solar mass over time.
    '''
    no_new_stars = []
    times = []
    previous = 0
    new = 0
    no_of_stars = []
    for i in file:
        new = len(i.m[(lower_limit<=i.m) & (upper_limit>=i.m)])
        no_new_stars.append(new-previous)
        previous = new
        times.append(i.t*978.461942384)
        no_of_stars.append(new)
    if all_stars == False:
        if plot == True and time == True:
            plt.plot(times,no_new_stars)
            plt.xlabel('Time[Myr]')
            plt.ylabel('Number of New Stars')
        elif plot == True and time == False:
            plt.plot(range(len(no_new_stars)),no_new_stars)
            plt.xlabel('Snapshot No')
            plt.ylabel('Number of New Stars')
        elif plot == False:
            return no_new_stars
    elif all_stars == True:
        if plot == True and time == True:
            plt.plot(times,no_of_stars)
            plt.xlabel('Time[Myr]')
            plt.ylabel('Number of Stars')
        elif plot == True and time == False:
            plt.plot(range(len(no_new_stars)),no_of_stars)
            plt.xlabel('Snapshot No')
            plt.ylabel('Number of Stars')
        elif plot == False:
            return no_of_stars

def average_star_age(file,plot = True,time = True):
    '''Average age (at every snapshot in Myr) of all stars'''
    average_ages = []
    for i in file:
        current_time = i.t*time_to_Myr
        ages = copy.copy(i.formation_time)
        ages = (current_time - ages*time_to_Myr)
        average_age = np.average(ages)
        average_ages.append(average_age)
    if plot ==True:
        plt.plot(times,average_ages)
    else:
        return average_ages
#Calculating the semi major axis for every possible configuration of these systems
def smaxis(system):
    '''Calculate the semi major axis (in m) between the secondary and primary in a system.'''
    k = system #Don't want to rewrite all the ks
    if k.no == 1: #Single star has no smaxis
        smaxis = 0
        return smaxis
    if len(k.m) == 2 and k.m[0] == k.m[1]:
        primary_id = k.ids[0]
        primary_mass = k.m[0]
        secondary_id = k.ids[1]
        secondary_mass = k.m[1]
        sec_ind = 1
    else:
        primary_id = k.primary_id
        primary_mass = k.primary
        secondary_id = np.array(k.ids)[k.m == k.secondary]
        if isinstance(secondary_id,np.ndarray):
            secondary_id = secondary_id[0]
        secondary_mass = 0
    for i in range(len(k.m)):
        if k.m[i] < primary_mass and k.m[i]> secondary_mass:
            secondary_mass = k.m[i]
            sec_ind = i
    if k.no == 2: #If there's two stars, only one possible semi major axis 
        vel_prim = k.v[np.argmax(k.m)]
        x_prim = k.x[np.argmax(k.m)]
        vel_sec = k.v[sec_ind]
        x_sec = k.x[sec_ind]
    if k.no == 3: #If there's three stars, only three configs: [[1,2],3] , [[1,3],2] and [[2,3],1]
        for i in k.structured_ids:
            if isinstance(i,list) and primary_id in i:
                if secondary_id in i:
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                else:
                    vel_prim = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.v)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.v)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                    x_prim = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.x)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.x)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                    primary_mass = (np.array(k.m)[np.array(k.ids) == i[0]]+np.array(k.m)[np.array(k.ids) == i[1]])[0]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
            elif isinstance(i,list) and secondary_id in i:
                if primary_id not in i:
                    vel_sec = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.v)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.v)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                    x_sec = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.x)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.x)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                    secondary_mass = np.array(k.m)[np.array(k.ids) == i[0]]+np.array(k.m)[np.array(k.ids) == i[1]]
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]

    if k.no == 4:# 4 is the most complex  [[1,2],[3,4]],[[1,3/4],[2,3/4]],[[[1,2],3/4],3/4],[[[1,3/4],2],3/4] or [[[1,3/4],3/4],2]
        struc_list = []
        for i in k.structured_ids:
            if isinstance(i,list):
                struc_list.append(len(i))
            else:
                struc_list.append(0)
        if sum(struc_list) == 4: #It is a binary of binaries
            for i in k.structured_ids:
                if primary_id in i and secondary_id in i:
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                else:
                    if primary_id in i and secondary_id not in i:
                        vel_prim = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.v)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.v)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                        x_prim = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.x)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.x)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                        primary_mass = (np.array(k.m)[np.array(k.ids) == i[0]]+np.array(k.m)[np.array(k.ids) == i[1]])
                    elif primary_id not in i and secondary_id in i:
                        vel_sec = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.v)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.v)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                        x_sec = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.x)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.x)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                        secondary_mass = np.array(k.m)[np.array(k.ids) == i[0]]+np.array(k.m)[np.array(k.ids) == i[1]]
        elif sum(struc_list) == 2: #It is hierarchial
            structure = []
            for i in k.structured_ids:
                substructure = []
                if isinstance(i,list):
                    for j in i:
                        if isinstance(j,list) and primary_id in j and secondary_id in j:
                            vel_prim = k.v[np.argmax(k.m)]
                            x_prim = k.x[np.argmax(k.m)]
                            vel_sec = k.v[sec_ind]
                            x_sec = k.x[sec_ind]
                        elif isinstance(j,list) and primary_id in j:
                            vel_prim = (np.array(k.m)[np.array(k.ids) == j[0]]*np.array(k.v)[np.array(k.ids) == j[0]] + np.array(k.m)[np.array(k.ids) == j[1]]*np.array(k.v)[np.array(k.ids) == j[1]])/(np.array(k.m)[np.array(k.ids) == j[1]]+np.array(k.m)[np.array(k.ids) == j[0]])
                            x_prim = (np.array(k.m)[np.array(k.ids) == j[0]]*np.array(k.x)[np.array(k.ids) == j[0]] + np.array(k.m)[np.array(k.ids) == j[1]]*np.array(k.x)[np.array(k.ids) == j[1]])/(np.array(k.m)[np.array(k.ids) == j[1]]+np.array(k.m)[np.array(k.ids) == j[0]])
                            primary_mass = np.array(k.m)[np.array(k.ids) == j[0]]+np.array(k.m)[np.array(k.ids) == j[1]]
                            substructure.append(42.0)
                        elif isinstance(j,list) and secondary_id in j:
                            vel_sec = (np.array(k.m)[np.array(k.ids) == j[0]]*np.array(k.v)[np.array(k.ids) == j[0]] + np.array(k.m)[np.array(k.ids) == j[1]]*np.array(k.v)[np.array(k.ids) == j[1]])/(np.array(k.m)[np.array(k.ids) == j[1]]+np.array(k.m)[np.array(k.ids) == j[0]])
                            x_sec = (np.array(k.m)[np.array(k.ids) == j[0]]*np.array(k.x)[np.array(k.ids) == j[0]] + np.array(k.m)[np.array(k.ids) == j[1]]*np.array(k.x)[np.array(k.ids) == j[1]])/(np.array(k.m)[np.array(k.ids) == j[1]]+np.array(k.m)[np.array(k.ids) == j[0]])
                            secondary_mass = np.array(k.m)[np.array(k.ids) == j[0]]+np.array(k.m)[np.array(k.ids) == j[1]]
                            substructure.append(24.0)
                        elif isinstance(j,list) and primary_id not in j and secondary_id not in j:
                            some_vel = (np.array(k.m)[np.array(k.ids) == j[0]]*np.array(k.v)[np.array(k.ids) == j[0]] + np.array(k.m)[np.array(k.ids) == j[1]]*np.array(k.v)[np.array(k.ids) == j[1]])/(np.array(k.m)[np.array(k.ids) == j[1]]+np.array(k.m)[np.array(k.ids) == j[0]])
                            some_x = (np.array(k.m)[np.array(k.ids) == j[0]]*np.array(k.x)[np.array(k.ids) == j[0]] + np.array(k.m)[np.array(k.ids) == j[1]]*np.array(k.x)[np.array(k.ids) == j[1]])/(np.array(k.m)[np.array(k.ids) == j[1]]+np.array(k.m)[np.array(k.ids) == j[0]])
                            some_mass = np.array(k.m)[np.array(k.ids) == j[0]]+np.array(k.m)[np.array(k.ids) == j[1]]
                            substructure.append(2.4)
                        elif isinstance(j,list) == False:
                            substructure.append(j)
                    structure.append(substructure)
                else:
                    structure.append(i)
            for stru in structure:
                if isinstance(stru,list) and 42.0 in stru and secondary_id in stru:
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                elif isinstance(stru,list) and 24.0 in stru and primary_id in stru:
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                elif isinstance(stru,list) and 42.0 in stru and secondary_id not in stru:
                    vel_prim = ((primary_mass*vel_prim)+((np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=42.0]])*(np.array(k.v)[np.array(k.ids)==np.array(stru)[np.array(stru)!=42.0]])))/(primary_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=42.0]])
                    x_prim = (primary_mass*x_prim + np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=42.0]]*np.array(k.x)[np.array(k.ids)==np.array(stru)[np.array(stru)!=42.0]])/(primary_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=42.0]])
                    primary_mass = primary_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=42.0]]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                elif isinstance(stru,list) and 24.0 in stru and primary_id not in stru:
                    vel_sec = (secondary_mass*vel_sec + np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=24.0]]*np.array(k.v)[np.array(k.ids)==np.array(stru)[np.array(stru)!=24.0]])/(secondary_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=24.0]])
                    x_sec = (secondary_mass*x_sec + np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=24.0]]*np.array(k.x)[np.array(k.ids)==np.array(stru)[np.array(stru)!=24.0]])/(secondary_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=24.0]])
                    secondary_mass = secondary_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=24.0]]
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                elif isinstance(stru,list) and 2.4 in stru and primary_id in stru:
                    vel_prim = ((some_mass*some_vel)+((np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])*(np.array(k.v)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])))/(some_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])
                    x_prim = (some_mass*some_x + np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]]*np.array(k.x)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])/(some_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                    primary_mass = some_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]]
                elif isinstance(stru,list) and 2.4 in stru and secondary_id in stru:
                    vel_sec = ((some_mass*some_vel)+((np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])*(np.array(k.v)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])))/(some_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])
                    x_sec = (some_mass*some_x + np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]]*np.array(k.x)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])/(some_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]])
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    secondary_mass = some_mass+np.array(k.m)[np.array(k.ids)==np.array(stru)[np.array(stru)!=2.4]]
                
    
    x_prim = list(flatten(x_prim))
    x_sec= list(flatten(x_sec))
    vel_prim = list(flatten(vel_prim))
    vel_sec= list(flatten(vel_sec))
    binding_energy = Binding_Energy(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
    mu = 1.9891e30*((primary_mass*secondary_mass)/(primary_mass+secondary_mass))
    eps = binding_energy/mu
    mu2 = 1.9891e30*6.67e-11*(primary_mass+secondary_mass)
    smaxis = - (mu2/(2*eps))
    if isinstance(smaxis,np.ndarray):
        smaxis = smaxis[0]
    return smaxis

def semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec):
    '''Calculate the semimajor axis(in m) from given parameters'''
    x_prim = list(flatten(x_prim))
    x_sec= list(flatten(x_sec))
    vel_prim = list(flatten(vel_prim))
    vel_sec= list(flatten(vel_sec))
    binding_energy = Binding_Energy(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
    mu = 1.9891e30*((primary_mass*secondary_mass)/(primary_mass+secondary_mass))
    eps = binding_energy/mu
    mu2 = 1.9891e30*6.67e-11*(primary_mass+secondary_mass)
    semiax = - (mu2/(2*eps))
    if isinstance(semiax,np.ndarray):
        semiax = semiax[0]
    return semiax

#Calculating the semi major axis for every possible configuration of these systems
def smaxis_all(system):
    '''Calculate the semimajor axis between all subsystems in a system'''
    k = system
    if k.no == 1: #Single star has no smaxis
        smaxis = 0
        return smaxis
    if len(k.m) == 2 and k.m[0] == k.m[1]:
        primary_id = k.ids[0]
        primary_mass = k.m[0]
        secondary_id = k.ids[1]
        secondary_mass = k.m[1]
        sec_ind = 1
    else:
        primary_id = k.primary_id
        primary_mass = k.primary
        secondary_id = np.array(k.ids)[k.m == k.secondary]
        if isinstance(secondary_id,np.ndarray):
            secondary_id = secondary_id[0]
        secondary_mass = 0
    for i in range(len(k.m)):
        if k.m[i] < primary_mass and k.m[i]> secondary_mass:
            secondary_mass = k.m[i]
            sec_ind = i
    if k.no == 3:
        tert_mass = k.m[k.m < secondary_mass][0]
        tert_id = np.array(k.ids)[k.m == tert_mass][0]
        tert_x = k.x[k.m == tert_mass][0]
        tert_v = k.v[k.m == tert_mass][0]
    elif k.no == 4:
        tert_mass = k.m[k.m < secondary_mass][0]
        tert_id = np.array(k.ids)[k.m == tert_mass][0]
        tert_x = k.x[k.m == tert_mass][0]
        tert_v = k.v[k.m == tert_mass][0]
        
        quart_mass = k.m[k.m < secondary_mass][1]
        quart_id = np.array(k.ids)[k.m == quart_mass][0]
        quart_x = k.x[k.m == quart_mass][0]
        quart_v = k.v[k.m == quart_mass][0]
        
    if k.no == 2: #If there's two stars, only one possible semi major axis 
        vel_prim = k.v[np.argmax(k.m)]
        x_prim = k.x[np.argmax(k.m)]
        vel_sec = k.v[sec_ind]
        x_sec = k.x[sec_ind]
        
        smaxis_count = 1
        
        semiax = semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
    if k.no == 3: #If there's three stars, only three configs: [[1,2],3] , [[1,3],2] and [[2,3],1]
        smaxis_count = 2
        for i in k.structured_ids:
            if isinstance(i,list) and primary_id in i:
                if secondary_id in i:
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                    semiax1 = semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
                    semiax2 = semi_major_axis(primary_mass+secondary_mass,tert_mass,(x_prim*primary_mass+x_sec*secondary_mass)/(primary_mass+secondary_mass),tert_x,(vel_prim*primary_mass+vel_sec*secondary_mass)/(primary_mass+secondary_mass),tert_v)
                    smaxis_count = 2
                else:
                    semiax1 = semi_major_axis(np.array(k.m)[np.array(k.ids) == i[0]][0],np.array(k.m)[np.array(k.ids) == i[1]][0],np.array(k.x)[np.array(k.ids) == i[0]][0],np.array(k.x)[np.array(k.ids) == i[1]][0],np.array(k.v)[np.array(k.ids) == i[0]][0],np.array(k.v)[np.array(k.ids) == i[1]][0])
                    smaxis_count = 2
                    vel_prim = (np.array(k.m)[np.array(k.ids) == i[0]][0]*np.array(k.v)[np.array(k.ids) == i[0]][0] + np.array(k.m)[np.array(k.ids) == i[1]][0]*np.array(k.v)[np.array(k.ids) == i[1]][0])/(np.array(k.m)[np.array(k.ids) == i[1]][0]+np.array(k.m)[np.array(k.ids) == i[0]][0])
                    x_prim = (np.array(k.m)[np.array(k.ids) == i[0]][0]*np.array(k.x)[np.array(k.ids) == i[0]][0] + np.array(k.m)[np.array(k.ids) == i[1]][0]*np.array(k.x)[np.array(k.ids) == i[1]][0])/(np.array(k.m)[np.array(k.ids) == i[1]][0]+np.array(k.m)[np.array(k.ids) == i[0]][0])
                    primary_mass = (np.array(k.m)[np.array(k.ids) == i[0]]+np.array(k.m)[np.array(k.ids) == i[1]])[0]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                    semiax2 = semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
            elif isinstance(i,list) and secondary_id in i:
                if primary_id not in i:
                    semiax1 = semi_major_axis(np.array(k.m)[np.array(k.ids) == i[0]][0],np.array(k.m)[np.array(k.ids) == i[1]][0],np.array(k.x)[np.array(k.ids) == i[0]][0],np.array(k.x)[np.array(k.ids) == i[1]][0],np.array(k.v)[np.array(k.ids) == i[0]][0],np.array(k.v)[np.array(k.ids) == i[1]][0])
                    smaxis_count = 2
                    vel_sec = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.v)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.v)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                    x_sec = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.x)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.x)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
                    secondary_mass = np.array(k.m)[np.array(k.ids) == i[0]][0]+np.array(k.m)[np.array(k.ids) == i[1]][0]
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    semiax2 = semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)

    if k.no == 4:# 4 is the most complex  [[1,2],[3,4]],[[1,3/4],[2,3/4]],[[[1,2],3/4],3/4],[[[1,3/4],2],3/4] or [[[1,3/4],3/4],2]
        smaxis_count = 3
        struc_list = []
        for i in k.structured_ids:
            if isinstance(i,list):
                struc_list.append(len(i))
            else:
                struc_list.append(0)
        if sum(struc_list) == 4: #It is a binary of binaries
            i = k.structured_ids[0]
            semiax1 = semi_major_axis(np.array(k.m)[np.array(k.ids) == i[0]][0],np.array(k.m)[np.array(k.ids) == i[1]][0],np.array(k.x)[np.array(k.ids) == i[0]],np.array(k.x)[np.array(k.ids) == i[1]],np.array(k.v)[np.array(k.ids) == i[0]],np.array(k.v)[np.array(k.ids) == i[1]])
            vel_prim = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.v)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.v)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
            x_prim = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.x)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.x)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
            primary_mass = (np.array(k.m)[np.array(k.ids) == i[0]]+np.array(k.m)[np.array(k.ids) == i[1]])
            i = k.structured_ids[1]
            semiax2 = semi_major_axis(np.array(k.m)[np.array(k.ids) == i[0]][0],np.array(k.m)[np.array(k.ids) == i[1]][0],np.array(k.x)[np.array(k.ids) == i[0]],np.array(k.x)[np.array(k.ids) == i[1]],np.array(k.v)[np.array(k.ids) == i[0]],np.array(k.v)[np.array(k.ids) == i[1]])
            vel_sec = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.v)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.v)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
            x_sec = (np.array(k.m)[np.array(k.ids) == i[0]]*np.array(k.x)[np.array(k.ids) == i[0]] + np.array(k.m)[np.array(k.ids) == i[1]]*np.array(k.x)[np.array(k.ids) == i[1]])/(np.array(k.m)[np.array(k.ids) == i[1]]+np.array(k.m)[np.array(k.ids) == i[0]])
            secondary_mass = np.array(k.m)[np.array(k.ids) == i[0]]+np.array(k.m)[np.array(k.ids) == i[1]]
            semiax3 = semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
        elif sum(struc_list) == 2: #It is a hierarchial system
            structure = []
            for i in k.structured_ids:
                substructure = []
                if isinstance(i,list):
                    for j in i:
                        if isinstance(j,list):
                            semiax1 = semi_major_axis(k.m[np.array(k.ids) == j[0]][0],k.m[np.array(k.ids) == j[1]][0],k.x[np.array(k.ids) == j[0]],k.x[np.array(k.ids) == j[1]],k.v[np.array(k.ids) == j[0]],k.v[np.array(k.ids) == j[1]])
                            primary_mass = k.m[np.array(k.ids) == j[0]][0]+k.m[np.array(k.ids) == j[1]][0]
                            vel_prim = (k.v[np.array(k.ids) == j[0]]*k.m[np.array(k.ids) == j[0]]+k.v[np.array(k.ids) == j[1]]*k.m[np.array(k.ids) == j[1]])/(k.m[np.array(k.ids) == j[0]][0]+k.m[np.array(k.ids) == j[1]][0])
                            x_prim = (k.x[np.array(k.ids) == j[0]]*k.m[np.array(k.ids) == j[0]]+k.x[np.array(k.ids) == j[1]]*k.m[np.array(k.ids) == j[1]])/(k.m[np.array(k.ids) == j[0]][0]+k.m[np.array(k.ids) == j[1]][0])
                        else:
                            outside_id = j
                    semiax2 = semi_major_axis(primary_mass,k.m[np.array(k.ids) == outside_id][0],x_prim,k.x[np.array(k.ids) == outside_id],vel_prim,k.v[np.array(k.ids) == outside_id])
                    x_prim = (primary_mass*x_prim+k.m[np.array(k.ids) == outside_id][0]*k.x[np.array(k.ids) == outside_id])/(primary_mass+k.m[np.array(k.ids) == outside_id][0])
                    vel_prim = (primary_mass*vel_prim+k.m[np.array(k.ids) == outside_id][0]*k.v[np.array(k.ids) == outside_id])/(primary_mass+k.m[np.array(k.ids) == outside_id][0])
                    primary_mass = primary_mass+k.m[np.array(k.ids) == outside_id][0] 
                else:
                    out_outside_id = i
                        
            semiax3 = semi_major_axis(primary_mass,k.m[np.array(k.ids) == out_outside_id][0],x_prim,k.x[np.array(k.ids) == out_outside_id],vel_prim,k.v[np.array(k.ids) == out_outside_id])
                       
    if smaxis_count == 1:
        return semiax
    elif smaxis_count == 2:
        return np.array([semiax1,semiax2])
    elif smaxis_count == 3:
        return np.array([semiax1,semiax2,semiax3])

#Getting the total masses, primary masses, smaxes and companion mass ratio. Also gets the target primary masses for 
#smaxes and companion mass ratios.
def primary_total_ratio_axis(systems,lower_limit = 0,upper_limit = 10000,all_companions = False,attribute = 'Mass Ratio'):
    '''
Returns a list of the property you chose for systems with primaries in a certain mass range.

Inputs
----------
systems : list of star system objects.
The systems in a certain snapshot to be looked at.

Parameters
----------
lower limit : int,float,optional
The lower limit of the primary mass range.

upper limit : int,float,optional
The upper limit of the primary mass range.
    
all_companions: bool,optional
Whether to include all companions or just the most massive (for mass ratio) or all subsystems or just the subsystems with the primary and secondary (Semi Major Axis)

attribute: string,optional
The attribute that you want. Choose from 'System Mass','Primary Mass','Mass Ratio' or 'Semi Major Axis'.

Returns
-------
attribute_distribution: list
The distribution of the property that you requested.

Example
-------
1) primary_total_ratio_axis(M2e4_C_M_2e7_systems[-1],attribute = 'System Mass')
Returns the mass of all multi star systems 

2) primary_total_ratio_axis(M2e4_C_M_2e7_systems[-1],attribute = 'Primary Mass')
Returns the mass of all primaries in multi star systems 

3) primary_total_ratio_axis(M2e4_C_M_2e7_systems[-1],attribute = 'Mass Ratio',all_companions = True,lower_limit = 0.7,upper_limit = 1.3)
Returns the mass ratio of all solar type companions.

1) primary_total_ratio_axis(M2e4_C_M_2e7_systems[-1],attribute = 'Semi Major Axis',all_companions = True,lower_limit = 0.7,upper_limit = 1.3)
Returns the semi major axis of all subsystems in the system.

    '''

    masses = []
    primary_masses = []
    mass_ratios = []
    semi_major_axes = []
    for i in systems:
        if i.no>1: #Make sure you only consider the multi star systems.
            masses.append(i.tot_m)
            primary_masses.append(i.primary)
            if lower_limit<=i.primary<=upper_limit:
                if all_companions == False:
                    semi_major_axes.append(smaxis(i))
                    mass_ratios.append(i.mass_ratio)
                elif all_companions == True: #If you want to look at all companions or subsystems.
                    semi_major_axes.append(smaxis_all(i))
                    for j in i.m:
                        if j!= i.primary:
                            mass_ratios.append(j/i.primary)
    if attribute == 'System Mass':
        return masses
    elif attribute == 'Primary Mass':
        return primary_masses
    elif attribute == 'Mass Ratio':
        return mass_ratios
    elif attribute == 'Semi Major Axis':
        return list(flatten(semi_major_axes))
    else:
        return None

#Multiplicity Fraction over different masses with a selection ratio of companions
def multiplicity_fraction(systems,mass_break = 2,selection_ratio = 0,attribute = 'Fraction',bins = 'continous'):
    '''
Returns the multiplicity fraction or multiplicity properties over a mass range.

Inputs
----------
systems : list of star system objects.
The systems in a certain snapshot to be looked at.

Parameters
----------
mass_break : int,float,optional
The log seperation in masses.

selection_ratio : int,float,optional
The minimum mass ratio of the companions.
    
attribute: string,optional
The attribute that you want. Choose from 'Fraction'(Primary No/(Primary No+ Single No)),'All Companions'(Primary No/(Primary No+Single No+Companion Number) or 'Properties'.

bins: string,optional
The type of bins that you want. Choose from 'continous' (evenly spaced in log space) or 'observer' (Duchene Krauss bins).

Returns
-------
logmasslist: list
The list of masses in logspace.

Multiplicity_Fraction_List or Single & Primary & Companion_Fractions_List: list
The list of multiplicity fraction or the 3 lists of primary,single or companion fractions.

mult_sys_count: int
The number of systems with more than one star (When returning multiplicity fraction).

sys_count:int
The number of systems (including single star systems)(When returning multiplicity fraction).

Example
-------
1) multiplicity_fraction(M2e4_C_M_2e7_systems[-1],attribute = 'Fraction',bins = 'observer') 
Returns the logmasslist, multiplicity fraction list, count of all multiple star systems and the number of all systems with the Duchene Krauss Bins.

2) multiplicity_fraction(M2e4_C_M_2e7_systems[-1],attribute = 'Properties',bins = 'continous') 
Returns the logmasslist, single star fraction list, primary star fraction list and the companion star fraction list.
    '''

    m = []
    state = []
    for i in systems:
        if len(i.m) == 1:
            m.append(i.m[0])
            state.append(0)
        elif i.no>1:
            for j in i.m:
                if j>=i.primary*selection_ratio and j != i.primary:
                    m.append(j)
                    state.append(2)
            m.append(i.primary)
            masses = np.array(i.m)
            if len(masses[masses>=selection_ratio*i.primary])>1:
                state.append(1)
            elif len(masses[masses>=selection_ratio*i.primary]) == 1:
                state.append(0)

    minmass= 0.08 # Because we dont want brown dwarfs
    maxmass = max(m)
    if bins == 'continous':
        logmasslist= np.linspace(np.log10(minmass),np.log10(maxmass+1),num = int((np.log10(maxmass+1)-np.log10(minmass))/(np.log10(mass_break))))
    elif bins == 'observer':
        #masslist = np.array([0.08,0.1,0.7,1.5,5,8,16,maxmass+1])
        masslist = np.array([0.08,0.1,0.7,1.5,5,16,maxmass+1])
        if maxmass<16:
            masslist = np.array([0.08,0.1,0.7,1.5,5,16])
        logmasslist = np.log10(masslist)
    primary_fraction = np.zeros_like(logmasslist)
    single_fraction = np.zeros_like(logmasslist)
    secondary_fraction = np.zeros_like(logmasslist)
    other_fraction = np.zeros_like(logmasslist)
    alternative_fraction = np.zeros_like(logmasslist)
    sys_count = np.zeros_like(logmasslist)
    mult_sys_count = np.zeros_like(logmasslist)
    ind = np.digitize(np.log10(m),logmasslist)
    bins = [[]]*len(logmasslist)
    for i in range(len(bins)):
        bins[i] = []
    for i in range(len(m)):
        bin_no = ind[i]-1 
        bins[bin_no].append(state[i])
    for i in range(len(bins)):
        primary_count = 0
        secondary_count = 0
        solo_count = 0
        for j in bins[i]:
            if j==0:
                solo_count = solo_count + 1
            elif j == 1:
                primary_count = primary_count + 1
            else:
                secondary_count = secondary_count + 1
        if len(bins[i])>0:
            primary_fraction[i] = primary_count/len(bins[i])
            single_fraction[i] = solo_count/len(bins[i])
            secondary_fraction[i] = secondary_count/len(bins[i])
        else:
            primary_fraction[i] = np.nan
            single_fraction[i] = np.nan
            secondary_fraction[i] = np.nan
        if primary_count+solo_count>0:
            other_fraction [i] = primary_count/(primary_count+solo_count)
            mult_sys_count[i] = primary_count
            sys_count[i] = primary_count+solo_count
        else:
            other_fraction[i] = np.nan
            mult_sys_count[i] = np.nan
            sys_count[i] = np.nan
        if primary_count+solo_count+secondary_count>0:
            alternative_fraction[i] = primary_count/(primary_count+solo_count+secondary_count)
        else:
            alternative_fraction[i] = np.nan
    if attribute == 'Fraction':
        return logmasslist,other_fraction,mult_sys_count,sys_count
    elif attribute == 'All Companions':
        return logmasslist,alternative_fraction
    elif attribute == 'Properties':
        return logmasslist,single_fraction,primary_fraction,secondary_fraction
    else:
        return None

#Multiplicity Frequency over different masses with a selection ratio
def multiplicity_frequency(systems,mass_break = 2,selection_ratio = 0,bins = 'continous'):
    '''
Returns the multiplicity freqeuncy over a mass range.

Inputs
----------
systems : list of star system objects.
The systems in a certain snapshot to be looked at.

Parameters
----------
mass_break : int,float,optional
The log seperation in masses.

selection_ratio : int,float,optional
The minimum mass ratio of the companions.
    
bins: string,optional
The type of bins that you want. Choose from 'continous' (evenly spaced in log space) or 'observer' (Moe-DiStefano bins).

Returns
-------
logmasslist: list
The list of masses in logspace.

multiplicity_frequency: list
The list of multiplicity frequencies.

companion_count: int
The number of companions.

sys_count:int
The number of systems (including single star systems).

Example
-------
multiplicity_frequency(M2e4_C_M_2e7_systems[-1],bins = 'observer') 
Returns the logmasslist, multiplicity frequency list, count of the number of companions and the number of all systems with the Moe DiStefano Bins.

    '''
    m = []
    companions = []
    for i in systems:
        throw = 0
        m.append(i.primary)
        if i.no>1:
            throw = len(np.array(i.m)[np.array(i.m)<=selection_ratio*i.primary])
        companions.append(i.no-1-throw)
    minmass= 0.08 # Because we dont want brown dwarfs
    maxmass= max(m)
    if bins == 'continous':
        logmasslist= np.linspace(np.log10(minmass),np.log10(maxmass),num = int((np.log10(maxmass)-np.log10(minmass))/(np.log10(mass_break))))
    elif bins == 'observer':
        logmasslist = np.log10(np.array([minmass,0.8,1.2,2.0,5.0,9.0,16.0,maxmass+1]))
        if maxmass<16:
            masslist = np.array([0.08,0.1,0.7,1.5,5,16])
            logmasslist = np.log10(masslist)
    ind = np.digitize(np.log10(m),logmasslist)
    bins = [[]]*len(logmasslist)
    for i in range(len(bins)):
        bins[i] = []
    for i in range(len(m)):
        bin_no = ind[i]-1 
        bins[bin_no].append(companions[i])
    multiplicity_frequency = np.zeros_like(bins)
    companion_count = np.zeros_like(bins)
    sys_count = np.zeros_like(bins)
    lognums = np.zeros_like(bins)
    for i in range(len(bins)):
        sys_count[i] = len(bins[i])
        companion_count[i] = sum(bins[i])
        if sys_count[i] == 0:
            multiplicity_frequency[i] = np.nan
        else:
            multiplicity_frequency[i] = sum(bins[i])/len(bins[i])
    return logmasslist,multiplicity_frequency,companion_count,sys_count

#This is the weighted probability sum of the chances of having the number of companions. This allows us to check if companions
#are randomly distributed or not
def randomly_distributed_companions(systems,file,snapshot,lower_limit = 1/1.5,upper_limit = 1.5,target_mass = 1,mass_ratio = np.linspace(0,1,num = 11),plot = True):
    '''
Returns the expected distribution of secondary companions if the drawing was random.

Inputs
----------
systems : list of star system objects.
The systems in a certain snapshot to be looked at.

file: list of sinkdata objects
The original file before system assignment.

snapshot: int
The snapshot that you want to look at

Parameters
----------
lower_limit : int,float,optional
The lower limit of the primary mass range.

upper_limit : int,float,optional
The upper limit of the primary mass range.

target_mass : int,float,optional
The target mass of primaries to cut the IMF at.
    
mass_ratio: range,list,array,optional
The bins for the mass ratios. Default is np.linspace(0,1,11)

plot: bool,optional
Whether to expected distribution or not

Returns
-------
Nsystems_with_M_companion_mass: array
The log number of companions expected at a certain mass ratio.

Stellar_Mass_PDF: array
The normalized log IMF until the primary mass

Example
-------
1) randomly_distributed_companions(M2e4_C_M_2e7_systems[-1],M2e4_C_M_2e7,-1,attribute = 'Fraction') 
Plots the expected companion distribution if the stars were randomly drawn from the IMF.
    '''
    systems_target = []
    count = 0
    for i in systems:
        if lower_limit<=i.primary<=upper_limit:
            systems_target.append(i)
        count += 1
    no2 = 0
    no3 = 0
    no4 = 0
    for i in systems_target:
        if i.no == 2:
            no2 = no2 +1
        elif i.no == 3:
            no3 = no3 + 1
        elif i.no == 4:
            no4 = no4 + 1
    w2 = no2/(no2+no3+no4)
    w3 = no3/(no2+no3+no4)
    w4 = no4/(no2+no3+no4)
    
    m_sorted = np.sort(file[snapshot].m[file[snapshot].m<(target_mass)])
    Nstars = len(m_sorted)
    P_full = np.arange(Nstars)/Nstars
    
    plot_masses = np.array(mass_ratio)*target_mass
    P_array = np.interp(plot_masses,m_sorted,P_full)
    
    Stellar_Mass_PDF = np.diff(P_array)*len(systems_target)
    
    P1 = w2* P_array + w3*(P_array)**2 + w4*(P_array)**3
    
    probability_M = np.diff(P1)
    
    Nsystems_with_M_companion_mass = probability_M*len(systems_target)
    
    Nsystems_with_M_companion_mass = np.insert(Nsystems_with_M_companion_mass,0,0)
    if plot == True:
        plt.step(mass_ratio,Nsystems_with_M_companion_mass)
        plt.yscale('log')
    else:
        return Nsystems_with_M_companion_mass,Stellar_Mass_PDF

#Describes the time evolution of the multiplicity fraction of different masses with two lines, one that
#shows the multiplicity at a given time and one that only chooses stars that remain solar mass
def Multiplicity_Fraction_Time_Evolution(file,Master_File,filename,steps=1,read_in_result = True,start = 0,target_mass = 1,upper_limit = 1.5,lower_limit = 1/1.5,plot = True):
    '''
Returns the evolution of the multiplicity fraction for a selected primary mass, either the fraction at a time or only for stars that dont accrete more.

Inputs
----------
file: list of sinkdata objects
The original file before system assignment.

Master_File: list of list of star system objects
All of the systems for the original file.

Parameters
----------
steps : int,optional
The number of snapshots to include in one step. By default, it is 1 meaning every snapshot.

read_in_result :bool,optional
Whether to read in results or perform system assignment for each snapshot.

start : bool,optional
First snapshot to look at. By default, it is the first snapshot with stars of the target mass.
    
target_mass: int,float,optional
The target primary mass to consider.

upper_limit: int,float,optional
The highest allowed mass for the primary.

lower_limit: int,float,optional
The lowest allowed mass for the primary.

Returns
-------
time: list
The times in the simulation (in free fall time).

fraction: list
The multiplicity fraction of target mass primaries at any time.

consistent_fraction: list
The multiplicity fraction of target mass primaries that stay the same mass at any time.

Example
-------
Multiplicity_Fraction_Time_Evolution(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,'M2e4_C_M_2e7') 
Plots the multiplicity time fraction over the runtime of the simulation.
    '''
    
    consistent_solar_mass = []
    consistent_solar_mass_unb = []
    if read_in_result == False:
        last_snap = system_creation(file,snapshot) #Getting the primaries in the last snap
        steps = steps
    elif read_in_result == True:
        last_snap = Master_File[-1]
        steps = 1
    #Getting a list of primaries that stay around the target mass at the end
    for i in last_snap:
        if lower_limit<=i.primary<=upper_limit and i.no>1:
            consistent_solar_mass.append(i.ids[list(i.m).index(i.primary)])
        elif i.no==1 and lower_limit<=i.m[0]<=upper_limit:
            consistent_solar_mass_unb.append(i.ids)
    fraction = [] #This fraction comes without ignoring the primaries that change mass
    fraction1 = [] #This fraction checks that the primaries are at a consistent mass
    masses = []
    time = []
    start = Mass_Creation_Finder(file,min_mass = lower_limit)
    #this one gets the masses and finishes off the graph of the consistent primaries
    for i in tqdm(range(start,len(file),steps),desc = 'By Snapshot',position=0):
        if read_in_result == False:
            sys = system_creation(file,i)
        elif read_in_result == True:
            sys = Master_File[i]
        primary_count = 0
        other_count = 0
        primary_easy = 0
        full_count = 0
        for j in sys:
            if lower_limit<=j.primary<=upper_limit:
                if j.no>1 and j.primary_id in consistent_solar_mass:
                    primary_count = primary_count + 1
                elif j.no == 1 and j.primary_id in consistent_solar_mass_unb:
                    other_count = other_count + 1
                full_count+=1
                if j.no >1:
                    primary_easy+=1
        if primary_count == 0 and other_count == 0:
            fraction1.append(np.nan)
        else:
            fraction1.append(primary_count/(primary_count+other_count))
        if primary_easy == 0 and full_count == 0:
            fraction.append(np.nan)
        else:
            fraction.append(primary_easy/full_count)
        time.append(file[i].t)
    time = np.array(time)
    ff_t = t_ff(file_properties(filename,param = 'm'),file_properties(filename,param = 'r'))
    time = (time/(ff_t*np.sqrt(file_properties(filename,param = 'alpha'))))
    if plot == True:
        plt.xlabel(r'Time $[\frac{t}{t_{ff}}]$')
        plt.ylabel('Multiplicity Fraction')
        plt.ylim([-0.1,1.1])
        plt.plot(time,fraction,label = 'Multiplicity Fraction for '+str(target_mass)+' Solar Mass Stars at any time')
        plt.plot(time,fraction1,label = 'Multiplicity Fraction for Stars that remain '+str(target_mass)+' solar mass')
        if target_mass == 1:
            plt.errorbar(max(time)-1,0.44,yerr=0.02,marker = 'o',capsize = 5,color = 'black',label = 'Observed Values')
        elif target_mass == 10:
            plt.errorbar(max(time)-1,0.6,lolims = True,yerr = 0.4,marker = 'o',capsize = 5,color = 'black',label = 'Observed Value')
        plt.legend(loc = (0.3,0.9))
        plt.text(0.5,0.1,'Target Mass ='+str(target_mass),transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
        plt.text(0.7,0.4,str(filename),transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
    elif plot == False:
        return time,fraction,fraction1

#Describes the time evolution of the multiplicity fraction of different masses with two lines, one that
#shows the multiplicity at a given time and one that only chooses stars that remain solar mass
def Multiplicity_Frequency_Time_Evolution(file,Master_File,filename,steps=1,read_in_result = True,start = 0,target_mass = 1,upper_limit = 1.5,lower_limit = 1/1.5,plot = True):
    '''
Returns the evolution of the multiplicity frequency for a selected primary mass, either the fraction at a time or only for stars that dont accrete more.

Inputs
----------
file: list of sinkdata objects
The original file before system assignment.

Master_File: list of list of star system objects
All of the systems for the original file.

Parameters
----------
steps : int,optional
The number of snapshots to include in one step. By default, it is 1 meaning every snapshot.

read_in_result :bool,optional
Whether to read in results or perform system assignment for each snapshot.

start : bool,optional
First snapshot to look at. By default, it is the first snapshot with stars of the target mass.
    
target_mass: int,float,optional
The target primary mass to consider.

upper_limit: int,float,optional
The highest allowed mass for the primary.

lower_limit: int,float,optional
The lowest allowed mass for the primary.

Returns
-------
time: list
The times in the simulation (in free fall time).

fraction: list
The multiplicity frequency of target mass primaries at any time.

consistent_fraction: list
The multiplicity frequency of target mass primaries that stay the same mass at any time.

Example
-------
Multiplicity_Frequency_Time_Evolution(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,'M2e4_C_M_2e7') 
Plots the multiplicity time frequency over the runtime of the simulation.
    '''
    consistent_solar_mass = []
    consistent_solar_mass_unb = []
    if read_in_result == False:
        last_snap = system_creation(file,snapshot) #Getting the primaries in the last snap
        steps = steps
    elif read_in_result == True:
        last_snap = Master_File[-1]
        steps = 1
    #Getting a list of primaries that stay around the target mass at the end
    for i in last_snap:
        if lower_limit<=i.primary<=upper_limit and i.no>1:
            consistent_solar_mass.append(i.primary_id)
        elif i.no==1 and lower_limit<=i.m[0]<=upper_limit:
            consistent_solar_mass_unb.append(i.ids)
    fraction = [] #This fraction comes without ignoring the primaries that change mass
    fraction1 = [] #This fraction checks that the primaries are at a consistent mass
    masses = []
    time = []
    start = Mass_Creation_Finder(file,min_mass = lower_limit)
    #this one gets the masses and finishes off the graph of the consistent primaries
    for i in tqdm(range(start,len(file),steps),desc = 'By Snapshot',position=0):
        if read_in_result == False:
            sys = system_creation(file,i)
        elif read_in_result == True:
            sys = Master_File[i]
        companion_count = 0
        other_count = 0
        companion_easy = 0
        full_count = 0
        for j in sys:
            if lower_limit<=j.primary<=upper_limit:
                if j.no>1 and j.primary_id in consistent_solar_mass:
                    companion_count = companion_count + j.no - 1
                elif j.no == 1 and j.primary_id in consistent_solar_mass_unb:
                    other_count = other_count + 1
                full_count+=1
                if j.no >1:
                    companion_easy+= j.no-1
        if companion_count == 0 and other_count == 0:
            fraction1.append(np.nan)
        else:
            fraction1.append(companion_count/(full_count))
        if companion_easy == 0 and full_count == 0:
            fraction.append(np.nan)
        else:
            fraction.append(companion_easy/full_count)
        time.append(file[i].t)
    time = np.array(time)
    ff_t = t_ff(file_properties(filename,param = 'm'),file_properties(filename,param = 'r'))
    time = (time/(ff_t*np.sqrt(file_properties(filename,param = 'alpha'))))
    if plot == True:
        plt.xlabel(r'Time $[\frac{t}{t_{ff}}]$')
        plt.ylabel('Multiplicity Frequency')
        plt.ylim([-0.1,3.1])
        plt.plot(time,fraction,label = 'Multiplicity Frequency for '+str(target_mass)+' Solar Mass Stars at any time')
        plt.plot(time,fraction1,label = 'Multiplicity Frequency for Stars that remain '+str(target_mass)+' solar mass')
        if target_mass == 1:
            plt.errorbar(max(time)-1,0.44,yerr=0.02,marker = 'o',capsize = 5,color = 'black',label = 'Observed Values')
        elif target_mass == 10:
            plt.errorbar(max(time)-1,0.6,lolims = True,yerr = 0.4,marker = 'o',capsize = 5,color = 'black',label = 'Observed Value')
        plt.legend(loc = (0.3,0.9))
        plt.text(0.7,0.1,'Target Mass ='+str(target_mass),transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
        plt.text(0.7,0.4,str(filename),transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
    else:
        return time,fraction,fraction1
def YSO_multiplicity(file,Master_File,min_age = 0,target_age = 2,start = 1000):
    '''
The multiplicity fraction of all objects in a certain age range.

Inputs
----------
file: list of sinkdata objects
The original file before system assignment.

Master_File: list of list of star system objects
All of the systems for the original file.

Parameters
----------
min_age : int,float,optional
The minimum age of objects.

target_age :int,float,optional
The maximum age of the objects

Returns
-------
multiplicity: list
The multiplicity fraction of the objects in the age range.

object_count: list
The number of objects in the age range.

average_mass: list
The average mass of the objects in the age range.

Example
-------
YSO_multiplicity(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems)
    '''    
    form = []
    for i in range(0,len(file)):
        if 'ProtoStellarAge' in file[i].extra_data_labels and file[i].val('ProtoStellarAge') is not None:
            form.append(-1)
        else:
            form.append(1)
    multiplicity = []
    bin_count = []
    average_mass = []
    for k in tqdm(range(len(Master_File)),position = 0):
        i = Master_File[k]
        current_time = file[k].t
        pcount = 0
        ubcount = 0
        tot_mass = 0
        for j in i:
            age_checker = 0
            prim_id = j.primary_id
            age = 0
            for Id in j.ids:
                if form[k] == -1:
                    age = (current_time-file[k].val('ProtoStellarAge')[file[k].id == Id])*time_to_Myr
                    if min_age<=age<=target_age:
                        age_checker += 1
                elif form[k] == 1:
                    first_snap = first_snap_finder(Id,file)
                    form_time = file[first_snap].formation_time[file[first_snap].id == Id]
                    age = (current_time - form_time)*time_to_Myr
                    if min_age<=age<=target_age:
                        age_checker += 1
            semaxis = smaxis(j)/m_to_AU
            if age_checker == j.no and j.no>1 and 20.0<=semaxis<=10000.0:
                pcount += 1
                tot_mass += j.primary 
            elif age_checker == j.no and j.no == 1:
                ubcount+= 1
                tot_mass += j.primary 
        if pcount+ubcount == 0:
            multiplicity.append(np.nan)
            average_mass.append(np.nan)
        else:
            multiplicity.append(pcount/(pcount+ubcount))
            average_mass.append(tot_mass/(pcount+ubcount))
        bin_count.append(pcount+ubcount) 
    return multiplicity,bin_count,average_mass

#This function tracks the evolution of different stars over their lifetime
def star_multiplicity_tracker(file,Master_File,T = 2,dt = 0.5,read_in_result = True,plot = False,target_mass = 1,upper_limit = 1.5,lower_limit = 1/1.5,zero = 'Consistent Mass',steps = 1,select_by_time = True,random_override = False,manual_random = False,sample_size = 20):
    '''
The status of stars born in a certain time range tracked throughout their lifetime in the simulation.

Inputs
----------
file: list of sinkdata objects
The original file before system assignment.

Master_File: list of list of star system objects
All of the systems for the original file.

Parameters
----------
T : int,float,optional
The time that the stars are born at.

dt :int,float,optional
The tolerance of the birth time. For example, if the simulation runs for 10 Myrs, T = 2 and dt = 0.5, it will choose stars born between 7.75 and 8.25 Myrs.

read_in_result: bool,optional
Whether to perform system assignment or use the already assigned system.

plot: bool,optional
Whether to return the times and multiplicities or plot them.

target_mass: int,float,optional
The target mass of primary to look at

upper_limit: int,float,optional
The upper limit of the target mass range

lower_limit: int,float,optional
The lower limit of the target mass range

steps: int,optional
The number of snapshots in one bin. If reading by result, this defaults to looking at every snapshot.

select_by_time: bool,optional:
Whether to track all stars or only those in a time frame.

random_override: bool,optional:
If you want to control a random sampling. By default, it does look at a random sample of over the sample size.

manual_random: bool,optional
Your choice to look at a random sample or not (only for plotting).

sample_size: int,optional
The amount of random stars to track (only to plot).

zero: string,optional
Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.

Returns
-------
all_times: list of lists
The times for each of the stars all in one list.

all_status: list of lists
The status of each of the stars at each time. If the status is -1, it is a companion, 0, it is single, otherwise it is a primary with status denoting the number of companions

ids: list
The id of each of the stars.

maturity_times: list
The time that each star stops accreting.

Tend:list
The end time for each star.

birth_times:list
The formation times for each star.

Example
-------
star_multiplicity_tracker(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems,T = 2,dt = 0.33)
    '''    
    consistent_solar_mass = []
    if read_in_result == False:
        last_snap = system_creation(file,-1) #Getting the primaries in the last snap
        steps = steps
    elif read_in_result == True:
        last_snap = Master_File[-1]
        steps = 1
    max_mass = 0
    min_mass = 0.08
    birth_times = []
    #Getting a list of primaries that stay around the target mass at the end
    for i in last_snap:
        if lower_limit<=i.primary<=upper_limit and i.no>1:
            consistent_solar_mass.append(i.primary_id)  
            if 'ProtoStellarAge' in file[-1].extra_data_labels:
                birth_time = (file[-1].t - file[-1].val('ProtoStellarAge')[file[-1].id == i.primary_id])*time_to_Myr
            else:
                first_snap = first_snap_finder(i.primary_id,file)
                birth_time = file[first_snap].t*time_to_Myr
            birth_times.append(birth_time)
        elif i.no==1 and lower_limit<=i.m[0]<=upper_limit:
            consistent_solar_mass.append(i.ids)
            if 'ProtoStellarAge' in file[-1].extra_data_labels:
                birth_time = (file[-1].t - file[-1].val('ProtoStellarAge')[file[-1].id == i.primary_id])*time_to_Myr
            else:
                first_snap = first_snap_finder(i.primary_id,file)
                birth_time = file[first_snap].t*time_to_Myr
            birth_times.append(birth_time)
    if select_by_time == True:
        Tend = file[-1].t*time_to_Myr
        kicked = 0 
        kept = 0
        og = len(consistent_solar_mass)
        copy = consistent_solar_mass.copy()
        for i in tqdm(copy,desc = 'Selecting By Maturity Time',position=0):
            if 'ProtoStellarAge' in file[-1].extra_data_labels:
                birth_time = file[-1].val('ProtoStellarAge')[file[-1].id == i]
            else:
                first_snap = first_snap_finder(i,file)
                birth_time = file[first_snap].t
            if birth_time*time_to_Myr>(Tend-T)+dt/2 or birth_time*time_to_Myr<(Tend-T)-dt/2:
                consistent_solar_mass.remove(i)
                kicked += 1
            else:
                kept += 1
        print('Kept = '+str(kept))
        print('Removed = '+str(kicked))
        print('Original Total = '+str(og))
    all_times = []
    all_status = []
    first_snaps = []
    change_in_status = []
    time_short = []
    maturity_times = []
    for i in tqdm(consistent_solar_mass,desc = 'Star of Interest',position=0):
        if zero == 'Consistent Mass':
            first_snap = first_snap_mass_finder(i,file,lower_limit,upper_limit)
        elif zero == 'Formation':
            first_snap = first_snap_finder(i,file)
        first_snaps.append(first_snap)
        birth_time = file[first_snap].t
        maturity_times.append(birth_time*time_to_Myr)
        times = []
        status = []
        for j in file:
            if j.t-birth_time>= 0:
                times.append((j.t-birth_time)*time_to_Myr)
        all_times.append(times)
        time_array = np.array(times)
        time_short.append(list((time_array[1:]+time_array[:-1])/2))
        statuses = []
        for k in range(first_snap,len(file),steps):
            status = 0
            if read_in_result == False:
                sys = system_creation(file,k)
            elif read_in_result == True:
                sys = Master_File[k]
            for l in sys:
                if l.no>1 and i in l.ids:
                    if i == l.primary_id:
                        status = l.no-1
                    else:
                        status = -1
                elif l.no == 1 and i == l.ids:
                    status = 0
            statuses.append(status)
        all_status.append(statuses)
        change_in_status.append(np.diff(statuses))
    ids = consistent_solar_mass
    if plot == True:
        if len(all_status)>sample_size:
            rand = True
        else:
            rand = False
        if random_override == True:
            rand = manual_random
        if rand == False:
            plt.figure(figsize=(10,10))
            plt.xlim(-0.01,max(flatten(all_times))*1.1)
            offset = np.linspace(-0.3,0.3,len(all_status))
            for i in range(len(all_status)):
                plt.plot(all_times[i],all_status[i]+offset[i],label = ids[i])
            plt.legend(loc = 'best',bbox_to_anchor=(1.1, 1, 0, 0))
            #plt.text(max(flatten(all_times))*0.8,-0.5,Files_key[n],fontsize = 12)
            plt.text(max(flatten(all_times))*0.8,-0.75,'Target Mass = '+str(target_mass),fontsize = 12)
            plt.xlabel('Time (in Myrs)')
            plt.ylabel('Status')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            #plt.savefig(Files_key[n]+'_'+str(target_mass)+'.png')
            plt.show()
            plt.figure(figsize=(10,10))
            plt.xlim(-0.01,max(flatten(time_short))*1.1)
            offset = np.linspace(-0.3,0.3,len(change_in_status))
            for i in range(len(change_in_status)):
                plt.plot(time_short[i],change_in_status[i]+offset[i],label = ids[i])
            plt.legend(loc = 'best',bbox_to_anchor=(1.1, 1, 0, 0))
            #plt.text(max(flatten(time_short))*0.8,-0.5,Files_key[n],fontsize = 12)
            plt.text(max(flatten(time_short))*0.8,-0.75,'Target Mass = '+str(target_mass),fontsize = 12)
            plt.xlabel('Time (in Myrs)')
            plt.ylabel('Change in status')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            #plt.savefig(Files_key[n]+'_'+str(target_mass)+'_Change.png')
            plt.show()
        elif rand == True:
            random_indices = random.sample(range(len(change_in_status)),sample_size)
            rand_times = []
            rand_status = []
            rand_ids = []
            for i in random_indices:
                rand_times.append(all_times[i])
                rand_status.append(all_status[i])
                rand_ids.append(ids[i])
            plt.figure(figsize = (20,20))
            plt.xlim(-0.01,max(flatten(rand_times))*1.1)
            offset = np.linspace(-0.3,0.3,len(rand_status))
            for i in range(len(rand_times)):
                plt.plot(rand_times[i],rand_status[i]+offset[i],label = rand_ids[i])
            plt.legend(loc = 'best',bbox_to_anchor=(1.1, 1, 0, 0))
            plt.xlabel('Time (in Myrs)')
            plt.ylabel('Status')
            #plt.text(max(flatten(rand_times))*0.8,-0.5,Files_key[n],fontsize = 12)
            plt.text(max(flatten(rand_times))*0.8,-0.75,'Target Mass = '+str(target_mass),fontsize = 12)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            #plt.savefig(Files_key[n]+'_'+str(target_mass)+'.png')
            plt.show()
            random_indices = random.sample(range(len(change_in_status)),sample_size)
            rand_times = []
            rand_status = []
            rand_ids = []
            for i in random_indices:
                rand_times.append(time_short[i])
                rand_status.append(change_in_status[i])
                rand_ids.append(ids[i])
            plt.figure(figsize = (20,20))
            plt.xlim(-0.01,max(flatten(rand_times))*1.1)
            offset = np.linspace(-0.3,0.3,len(rand_status))
            for i in range(len(rand_times)):
                plt.plot(rand_times[i],rand_status[i]+offset[i],label = rand_ids[i])
            plt.legend(loc = 'best',bbox_to_anchor=(1.1, 1, 0, 0))
            plt.xlabel('Time (in Myrs)')
            plt.ylabel('Change in Status')
            #plt.text(max(flatten(rand_times))*0.8,-0.5,Files_key[n],fontsize = 12)
            plt.text(max(flatten(rand_times))*0.8,-0.75,'Target Mass = '+str(target_mass),fontsize = 12)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            #plt.savefig(Files_key[n]+'_'+str(target_mass)+'_Change.png')
            plt.show()

    if plot == False:
        placeholder = 0
        if select_by_time == True:
            placeholder = Tend
        return all_times,all_status,ids,maturity_times,placeholder,birth_times

#This function gives the multiplicity fraction at different ages
def multiplicity_frac_and_age(file,Master_File,T = 2,dt = 0.5,target_mass = 1,upper_limit = 1.5,lower_limit = 1/1.5,read_in_result = True,select_by_time = True,zero = 'Formation',plot = True,steps = 1):
    '''
The average multiplicity fraction of stars born in a certain time range tracked throughout their lifetime in the simulation.

Inputs
----------
file: list of sinkdata objects
The original file before system assignment.

Master_File: list of list of star system objects
All of the systems for the original file.

Parameters
----------
T : int,float,optional
The time that the stars are born at.

dt :int,float,optional
The tolerance of the birth time. For example, if the simulation runs for 10 Myrs, T = 2 and dt = 0.5, it will choose stars born between 7.75 and 8.25 Myrs.

target_mass: int,float,optional
The target mass of primary to look at

upper_limit: int,float,optional
The upper limit of the target mass range

lower_limit: int,float,optional
The lower limit of the target mass range

read_in_result: bool,optional
Whether to perform system assignment or use the already assigned system.

select_by_time: bool,optional:
Whether to track all stars or only those in a time frame.

zero: string,optional
Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.

plot: bool,optional
Whether to return the times and multiplicities or plot them.

steps: int,optional
The number of snapshots in one bin. If reading by result, this defaults to looking at every snapshot.

Returns
-------
age_bins: array
The age over which the stars are in.

multiplicity: array
The average multiplicity fraction of the objects in the bins.

birth_times:list
The birth times of the stars.

Example
-------
multiplicity_frac_and_age(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems)
    '''  
    times,status,ids,maturity_times,Tend,birth_times = star_multiplicity_tracker(file,Master_File,T = T,dt = dt,read_in_result = read_in_result,plot = False,target_mass = target_mass,upper_limit=upper_limit,lower_limit=lower_limit,zero = zero,steps = steps,select_by_time=select_by_time)
    counted_all = []
    is_primary_all = []
    time_all = []; status_all = []
    lengths = []
    #plt.figure(figsize = (15,10))
    #for t,s in zip(times,status):
    #    plt.plot(t,s)
    for t,s in zip(times,status):
        time_all += t;status_all += s
        lengths.append(len(t))
    time_all = np.array(time_all);status_all = np.array(status_all)
    age_bins=np.linspace(0,max(time_all),max(lengths)+1)
    counted_all = status_all.copy();counted_all[status_all>=0] = 1;counted_all[status_all<0] = 0
    is_primary_all = status_all.copy();is_primary_all[status_all>0] = 1;is_primary_all[status_all<=0] = 0
    counted_in_bin, temp = np.histogram(time_all, bins=age_bins, weights=counted_all)
    is_prmary_in_bin, temp = np.histogram(time_all, bins=age_bins, weights=is_primary_all)
    multiplicity_in_bin = (is_prmary_in_bin)/(counted_in_bin)
    age_bins_mean = (age_bins[1:] + age_bins[:-1])/2
    times = []
    for i in file:
        times.append(i.t*time_to_Myr)
    if plot == True:
        if select_by_time == True:
            plt.figure()
            new_stars_count(file)
            plt.fill_between([max(times)-(T-dt/2),max(times)-(T+dt/2)],16,alpha = 0.3)
            plt.xlabel('Simulation Time [Myr]')
            plt.ylabel('No of New Stars')
            plt.show()
            plt.figure()
            new_stars_count_mass(file,lower_limit=lower_limit,upper_limit=upper_limit)
            plt.fill_between([max(times)-(T-dt/2),max(times)-(T+dt/2)],8,-2,alpha = 0.3)
            plt.xlabel('Simulation Time [Myr]')
            plt.ylabel('Change in # of Target Mass Stars')
            plt.show()
            plt.figure()
            plt.plot(age_bins_mean[age_bins_mean<(T-dt/2)],multiplicity_in_bin[age_bins_mean<(T-dt/2)])
        else:
            plt.plot(age_bins_mean,multiplicity_in_bin)
        #plt.plot(age_bins_mean,multiplicity_in_bin,label = 'Multiplicity at Age Plot')
        plt.ylim([-0.1,1.1])
        plt.xlabel('Age in Myrs')
        plt.ylabel('Average Multiplicity Fraction')
        #plt.text(0.1,0.8,Files_key[n],transform = plt.gca().transAxes)
        plt.text(0.1,0.7,'Target Mass ='+str(target_mass),transform = plt.gca().transAxes)
        #plt.legend()
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.show()
        plt.figure()
        if select_by_time == True:
            plt.plot(age_bins_mean[age_bins_mean<(T-dt/2)],(counted_in_bin)[age_bins_mean<(T-dt/2)])
        else:
            plt.plot(age_bins_mean,(counted_in_bin))
        plt.xlabel('Age in Myrs')
        plt.ylabel('Number of Stars')
        #plt.legend()
        plt.show()
    else:
        return age_bins_mean[age_bins_mean<(T-dt/2)],multiplicity_in_bin[age_bins_mean<(T-dt/2)],birth_times
    #return age_bins_mean,multiplicity_in_bin

#This function gives the multiplicity fraction at different ages
def multiplicity_freq_and_age(file,Master_File,T = 2,dt = 0.5,target_mass = 1,upper_limit = 1.5,lower_limit = 1/1.5,read_in_result = True,select_by_time = True,zero = 'Formation',plot = True,steps = 1):
    '''
The average multiplicity frequency of stars born in a certain time range tracked throughout their lifetime in the simulation.

Inputs
----------
file: list of sinkdata objects
The original file before system assignment.

Master_File: list of list of star system objects
All of the systems for the original file.

Parameters
----------
T : int,float,optional
The time that the stars are born at.

dt :int,float,optional
The tolerance of the birth time. For example, if the simulation runs for 10 Myrs, T = 2 and dt = 0.5, it will choose stars born between 7.75 and 8.25 Myrs.

target_mass: int,float,optional
The target mass of primary to look at

upper_limit: int,float,optional
The upper limit of the target mass range

lower_limit: int,float,optional
The lower limit of the target mass range

read_in_result: bool,optional
Whether to perform system assignment or use the already assigned system.

select_by_time: bool,optional:
Whether to track all stars or only those in a time frame.

zero: string,optional
Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.

plot: bool,optional
Whether to return the times and multiplicities or plot them.

steps: int,optional
The number of snapshots in one bin. If reading by result, this defaults to looking at every snapshot.

Returns
-------
age_bins: array
The age over which the stars are in.

multiplicity_frequency: array
The average multiplicity frequency of the objects in the bins.

Example
-------
multiplicity_freq_and_age(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems)
    '''  
    times,status,ids,maturity_times,Tend,birth_times = star_multiplicity_tracker(file,Master_File,T = T,dt = dt,read_in_result = read_in_result,plot = False,target_mass = target_mass,upper_limit=upper_limit,lower_limit=lower_limit,zero = zero,steps = steps,select_by_time=select_by_time)
    counted_all = []
    is_primary_all = []
    time_all = []; status_all = []
    lengths = []
    for t,s in zip(times,status):
        time_all += t;status_all += s
        lengths.append(len(t))
    time_all = np.array(time_all);status_all = np.array(status_all)
    age_bins=np.linspace(0,max(time_all),max(lengths)+1)
    counted_all = status_all.copy();counted_all[status_all>=0] = 1;counted_all[status_all<0] = 0
    is_primary_all = status_all.copy();is_primary_all[status_all<=0] = 0
    counted_in_bin, temp = np.histogram(time_all, bins=age_bins, weights=counted_all)
    is_prmary_in_bin, temp = np.histogram(time_all, bins=age_bins, weights=is_primary_all)
    multiplicity_in_bin = (is_prmary_in_bin)/(counted_in_bin)
    age_bins_mean = (age_bins[1:] + age_bins[:-1])/2
    times = []
    for i in file:
        times.append(i.t*time_to_Myr)
    if plot == True:
        if select_by_time == True:
            plt.figure()
            new_stars_count(file)
            plt.fill_between([max(times)-(T-dt/2),max(times)-(T+dt/2)],16,alpha = 0.3)
            plt.xlabel('Simulation Time [Myr]')
            plt.ylabel('No of New Stars')
            plt.show()
            plt.figure()
            new_stars_count_mass(file,lower_limit=lower_limit,upper_limit=upper_limit)
            plt.fill_between([max(times)-(T-dt/2),max(times)-(T+dt/2)],8,-2,alpha = 0.3)
            plt.xlabel('Simulation Time [Myr]')
            plt.ylabel('Change in # of Target Mass Stars')
            plt.show()
            plt.figure()
            plt.plot(age_bins_mean[age_bins_mean<(T-dt/2)],multiplicity_in_bin[age_bins_mean<(T-dt/2)])
        else:
            plt.plot(age_bins_mean,multiplicity_in_bin)
        #plt.plot(age_bins_mean,multiplicity_in_bin,label = 'Multiplicity at Age Plot')
        plt.ylim([-0.1,3.1])
        plt.xlabel('Age in Myrs')
        plt.ylabel('Average Multiplicity Frequency')
        #plt.text(0.1,0.8,Files_key[n],transform = plt.gca().transAxes)
        plt.text(0.1,0.7,'Target Mass ='+str(target_mass),transform = plt.gca().transAxes)
        #plt.legend()
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.show()
        plt.figure()
        if select_by_time == True:
            plt.plot(age_bins_mean[age_bins_mean<(T-dt/2)],(counted_in_bin)[age_bins_mean<(T-dt/2)])
        else:
            plt.plot(age_bins_mean,(counted_in_bin))
        plt.xlabel('Age in Myrs')
        plt.ylabel('Number of Stars')
        #plt.legend()
        plt.show()
    else:
        return age_bins_mean[age_bins_mean<(T-dt/2)],multiplicity_in_bin[age_bins_mean<(T-dt/2)],birth_times
    #return age_bins_mean,multiplicity_in_bin

def Orbital_Plot_2D(system,plot = True):
    '''Create an orbital plane projection plot of any system'''
    if system.no>1:
        #Getting the velocity and coordinates of the CoM (of only secondary and primary)
        com_coord = (system.primary*system.x[np.array(system.ids) == system.primary_id]+system.secondary*system.x[system.m == system.secondary])/(system.primary+system.secondary)
        com_vel = (system.primary*system.v[np.array(system.ids) == system.primary_id]+system.secondary*system.v[system.m == system.secondary])/(system.primary+system.secondary)
        #Getting the mass, coordiantes and velocity in the CoM frame
        m1 = system.primary
        m2 = system.secondary
        r1 = system.x[np.array(system.ids) == system.primary_id] - com_coord
        r2 = system.x[np.array(system.m) == system.secondary] - com_coord
        v1 = system.v[np.array(system.ids) == system.primary_id] - com_vel
        v2 = system.v[np.array(system.m) == system.secondary] - com_vel
        #Calculating the angular momentum and normalizing it
        L = m1*(np.cross(r1,v1)) + m2*(np.cross(r2,v2)) #Check with x and y
        L = L[0]
        l = np.linalg.norm(L)
        L_unit = L/l
        #Finding the two unit vectors
        unit_vector = [1,0,0]
        if L_unit[0]>0 and L_unit[1] == 0 and L_unit[2] == 0:
            unit_vector = [0,1,0]
        e1_nonnorm = np.cross(L_unit,unit_vector) #Check the 0th component
        e1 = e1_nonnorm/np.linalg.norm(e1_nonnorm) #Check if this is proper (Dont use ex or ey)
        e2 = np.cross(e1,L_unit) #Check that it is a proper shape
        #Getting the CoM of the whole system coordinates
        com_cord_all_nonnorm = 0
        com_vel_all_nonnorm = 0
        for i in range(system.no):
            com_cord_all_nonnorm += system.m[i]*system.x[i]
            com_vel_all_nonnorm += system.m[i]*system.v[i]
        com_cord_all = com_cord_all_nonnorm/sum(system.m)
        com_vel_all = com_vel_all_nonnorm/sum(system.m)
        #Now getting the x and v in the CoM frame
        com_frame_x = system.x - com_cord_all
        com_frame_v = system.v - com_vel_all
        #Finally, we project the x and v to the orbital plane
        x_new = np.zeros((system.no,2))
        v_new = np.zeros((system.no,2))
        for i in range(system.no):
            x_new[i][0] = np.dot(com_frame_x[i],e1)
            x_new[i][1] = np.dot(com_frame_x[i],e2)
            v_new[i][0] = np.dot(com_frame_v[i],e1)
            v_new[i][1] = np.dot(com_frame_v[i],e2)
        #Now we plot onto a quiver plot 
        if plot == True:
            plt.figure(figsize = (10,10))
            for i in range(system.no):
                plt.quiver(x_new[i][0]*pc_to_AU,x_new[i][1]*pc_to_AU,v_new[i][0],v_new[i][1])
                plt.scatter(x_new[i][0]*pc_to_AU,x_new[i][1]*pc_to_AU,s = system.m[i]*100)
            plt.xlabel('Coordinate 1 [AU]')
            plt.ylabel('Coordinate 2 [AU]')
            plt.show()
        #Checking original KE
        oKE = 0
        for i in range(system.no):
            oKE += 0.5*system.m[i]*np.linalg.norm(com_frame_v[i])**2
        #Checking new KE
        com_new_vel_all_nonnorm = 0
        for i in range(system.no):
            com_new_vel_all_nonnorm += system.m[i]*v_new[i]
        com_new_vel = np.array(com_new_vel_all_nonnorm)/sum(system.m)
        v_new_com_frame = np.zeros((system.no,2))
        for i in range(system.no):
            v_new_com_frame[i] = v_new[i] - com_new_vel
        nKE = 0
        for i in range(system.no):
            nKE += 0.5*system.m[i]*np.linalg.norm(v_new_com_frame[i])**2
        if plot == True:
            print("1 - KE'/KE = "+str((1-nKE/oKE).round(2)))
            print('Semi Major Axis of system is ' +str((smaxis(system)/m_to_AU).round(2))+ ' AU')
        if plot == False:
            return 1-nKE/oKE
    elif system.no == 1:
        if plot == True:
            print('No, this is a plot of one image')
        else:
            return np.nan

def smaxis_tracker(file,Master_File,system_ids,plot = True,KE_tracker = False):
    '''
Tracking the semi-major axis between some ids throughout the simulation runtime.
Inputs
----------
file: list of sinkdata objects
The original file before system assignment.

Master_File: list of list of star system objects
All of the systems for the original file.

system_ids: list
The ids to track the semi-major axis of.

Parameters
----------
plot: bool,optional
Whether to return the values or plot them.

KE_tracker: bool,optional
Whether to also look at the loss in kinetic energy from the orbital plane projection.

Returns
-------
smaxes: list
The semi major axis of the given ids throughout the simulation

no_of_stars: list
The number of stars in the system throughout the simulation

KE_tracks: list
The loss in KE from an orbital projection throughout time. Only returns this if KE_tracker is true.

times: list
The times that the system exists in the simulation.

Example
-------
smaxis_tracker(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems,[112324.0,1233431.0])
'''  

    smaxes = []
    times = []
    no_of_stars = []
    f_tracks = []
    for i in range(len(Master_File)):
        marker = 0
        times.append(file[i].t*time_to_Myr)
        for j in Master_File[i]:
            if(set(system_ids).issubset(set(j.ids))):
                smaxes.append(np.log10(smaxis(j)/m_to_AU))
                no_of_stars.append(j.no)
                if KE_tracker == True:
                    f_tracks.append(Orbital_Plot_2D(j,plot = False))
                marker = 1
        if marker == 0:
            smaxes.append(np.nan)
            no_of_stars.append(np.nan)
            f_tracks.append(np.nan)

    if plot == True:
        plt.figure(figsize = (10,10))
        plt.plot(times,smaxes)
        plt.xlabel('Time (Myr)')
        plt.ylabel('Log Semi Major Axis (AU)')
        plt.show()

        plt.figure(figsize = (10,10))
        plt.plot(times,no_of_stars)
        plt.xlabel('Time (Myr)')
        plt.ylabel('No of Stars in System')
        plt.show()
        
        if KE_tracker == True:
            plt.figure(figsize = (10,10))
            plt.plot(times,f_tracks)
            plt.xlabel('Time (Myr)')
            plt.ylabel('Kinetic Energy Loss in Orbital Projection')
            plt.show()
            
        
    elif plot == False:
        if KE_tracker == False:
            return smaxes,no_of_stars,times
        elif KE_tracker == True:
            return smaxes,no_of_stars,f_tracks,times

def formation_distance(id_list,file_name,log = True):
    '''The formation distance between two ids with the original file name provided as a string.'''
    pickle_file = open(file_name +'.pickle','rb')
    Brown_Dwarf_File = pickle.load(pickle_file)
    pickle_file.close()
    
    logdist = []
    first_snap1 = first_snap_finder(id_list[0],Brown_Dwarf_File)
    first_snap2 = first_snap_finder(id_list[1],Brown_Dwarf_File)
    first_snap_both = max([first_snap1,first_snap2])
    pos1 = Brown_Dwarf_File[first_snap_both].x[Brown_Dwarf_File[first_snap_both].id == id_list[0]]
    pos2 = Brown_Dwarf_File[first_snap_both].x[Brown_Dwarf_File[first_snap_both].id == id_list[1]]

    distance = np.linalg.norm(pos1-pos2)*206264.806
    if log == True:
        return np.log10(distance)
    else:
        return distance

def q_with_formation(Master_File,file_name,snapshot,limit = 10000,upper_mass_limit = 1.3,lower_mass_limit = 0.7):
    '''
Seperating the mass ratios based on formation distance.
Inputs
----------
Master_File: list of list of star system objects
All of the systems for the original file.

file_name: str
The name of the file to check the formation distance from.

snapshot: int
The snapshot to check.

Parameters
----------
limit: int,float,optional
The formation distance limit that you want to split by.

upper_mass_limit: int,float,optional
The upper mass limit for the primaries

lower_mass_limit: int,float,optional
The lower mass limit for the primaries

Returns
-------
q_list_under: list
The mass ratios under the formation distance limit

distance_list_under: list
The formation distance distribution under the formation distance limit. 

q_list_over: list
The mass ratios over the formation distance limit

distance_list_over: list
The formation distance distribution over the formation distance limit. 

all_dist: list
The formation distance for everything.

'''  
    q_list_under = []
    distance_list_under = []
    q_list_over = []
    distance_list_over = []
    all_dist = []
    for i in Master_File[snapshot]:
        if i.no > 1 and lower_mass_limit<=i.primary<=upper_mass_limit:
            for ids in i.ids:
                if ids != i.primary_id :
                    form_dist = formation_distance([ids,i.primary_id],file_name,log = False)
                    all_dist.append(form_dist)
                    if form_dist <= limit:
                        q_list_under.append(i.m[np.array(i.ids) == ids]/i.primary)
                        distance_list_under.append(form_dist)
                    if form_dist > limit:
                        q_list_over.append(i.m[np.array(i.ids) == ids]/i.primary)
                        distance_list_over.append(form_dist)
    return list(flatten(q_list_under)),distance_list_under,list(flatten(q_list_over)),distance_list_over,all_dist

#Using np.hist and moving the bins to the center of each bin
def hist(x,bins = 'auto',log =False,shift = False):
    '''
Create a histogram
Inputs
----------
x: data
The data to be binned

Parameters
----------
bins: int,list,str
The bins to use.

log: bool,optional
Whether to return number of objects in bin or log number of objects.

shift: bool,optional
Whether to shift the bins to the center or not.

Returns
-------
x_vals: list
The bins.

weights:list
The weights of each bin

Example
-------
hist(x)
'''  
    if x is None:
        return None,None
    if log == True:
        weights,bins = np.histogram(np.log10(x),bins = bins)
    elif log == False:
        weights,bins = np.histogram(x,bins = bins)
    if shift == True:
        xvals = (bins[:-1] + bins[1:])/2
    else:
        xvals = bins
    return xvals,weights

Plots_key = ['System Mass','Primary Mass','Mass Ratio','Semi Major Axis','Multiplicity','Multiplicity Time Evolution',
'Multiplicity Lifetime Evolution','YSO Multiplicity','Semi-Major Axis vs q']

def multiplicity_and_age_combined(file,Master_File,T_list,dt_list,upper_limit=1.3,lower_limit = 0.7,target_mass = 1,zero = 'Formation',multiplicity = 'Fraction',filename = None):
    '''
The average multiplicity of stars born in certain time ranges tracked throughout their lifetime in the simulation.

Inputs
----------
file: list of sinkdata objects
The original file before system assignment.

Master_File: list of list of star system objects
All of the systems for the original file.

Parameters
----------
T : list,optional
The time that the stars are born at.

dt :list,optional
The tolerance of the birth time.

target_mass: int,float,optional
The target mass of primary to look at

upper_limit: int,float,optional
The upper limit of the target mass range

lower_limit: int,float,optional
The lower limit of the target mass range

read_in_result: bool,optional
Whether to perform system assignment or use the already assigned system.

select_by_time: bool,optional:
Whether to track all stars or only those in a time frame.

zero: string,optional
Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.

plot: bool,optional
Whether to return the times and multiplicities or plot them.

steps: int,optional
The number of snapshots in one bin. If reading by result, this defaults to looking at every snapshot.

Returns
-------
age_bins: array
The age over which the stars are in.

multiplicity: array
The average multiplicity fraction of the objects in the bins.

Example
-------
multiplicity_frac_and_age(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems)
    '''  
    time_list = []
    mul_list = []
    for i in range(len(T_list)):
        if multiplicity == 'Fraction':
            time,mul,birth_times = multiplicity_frac_and_age(file,Master_File,T_list[i],dt_list[i],zero = zero,upper_limit=1.3,lower_limit = 0.7,target_mass = 1,plot = False)
        elif multiplicity == 'Frequency':
            time,mul,birth_times = multiplicity_freq_and_age(file,Master_File,T_list[i],dt_list[i],zero = zero,upper_limit=1.3,lower_limit = 0.7,target_mass = 1,plot = False)
        time_list.append(time)
        mul_list.append(mul)
    times = []
    for i in file:
        times.append(i.t*time_to_Myr)
    birth_times = np.array(birth_times)
    times,new_stars_co = hist(birth_times,bins = times)
    times = np.array(times)
    plt.plot((times[1:]+times[:-1])/2,new_stars_co)
    for i in range(len(T_list)):
        plt.fill_between([(times[-1]-T_list[i])-dt_list[i]/2,(times[-1]-T_list[i])+dt_list[i]/2],0,max(new_stars_co),alpha  = 0.3,label = 'T = '+str(T_list[i]))
    plt.legend()
    plt.xlabel('Age [Myr]')
    plt.ylabel('Number of New Stars')
    adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
    plt.figure(figsize = (10,10))
    for i in range(len(time_list)):
        plt.plot(time_list[i],np.array(mul_list[i]),label = str(T_list[i])+' Myr')
    if target_mass == 1:
        if multiplicity == 'Fraction':
            plt.errorbar(max(list(flatten(time_list))),0.44,yerr=0.02,marker = 'o',capsize = 5,color = 'black',label = 'Observed Values')
        elif multiplicity == 'Frequency':
            plt.errorbar(max(list(flatten(time_list))),0.5,yerr=0.04,marker = 'o',capsize = 5,color = 'black',label = 'Observed Values')
    elif target_mass == 10:
        if multiplicity == 'Fraction':
            plt.errorbar(max(list(flatten(time_list))),0.6,yerr=0.2,lolims = True,marker = 'o',capsize = 5,color = 'black',label = 'Observed Value')
        elif multiplicity == 'Frequency':
            plt.errorbar(max(list(flatten(time_list))),1.6,yerr=0.2,lolims = True,marker = 'o',capsize = 5,color = 'black',label = 'Observed Value')
        
    plt.legend()

    plt.xlabel('Age [Myr]')
    plt.ylabel('Multiplicity Fraction')
    if multiplicity == 'Fraction':
        plt.ylim([-0.05,1.05])
    elif multiplicity == 'Frequency':
        plt.ylim([-0.05,3.05])
        plt.ylabel('Multiplicity Frequency')
    plt.text(max(list(flatten(time_list)))/2,0.8,'Star Mass = $1 M_\odot$')
    if filename is not None:
        plt.text(max(list(flatten(time_list)))/2,0.5,filename)
    adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
    plt.show()
    
def One_Snap_Plots(which_plot,systems,file,filename = None,snapshot = None,upper_limit = 1.3,lower_limit = 0.7,target_mass = None,all_companions = True,bins = 10,log = True,compare = False,plot = True,read_in_result = True,filtered = False,filter_snaps_no = 10,min_q = 0.1,Master_File = None):
    '''
Create the plots for one snapshot
Inputs
----------
which_plot: string
The plot to be made.

systems: list of star system objects
The systems you want to analyze (1 snapshot or a filtered snapshot).

file:list of sinkdata objects
The original sinkdata file.

Parameters
----------
filename:string
The name of the original file. It will be labelled on the plot if provided.

snapshot: int,float
The snapshot number you want to look at. Only required for IMF comparisons and filter on.

target_mass: int,float,optional
The mass of the primaries of the systems of interest.

upper_limit: int,float,optional
The upper mass limit of the primaries of systems of interest.

lower_limit: int,float,optional
The lower mass limit of the primaries of systems of interest.

all_companions: bool,optional
Whether to include all companions in the mass ratio or semi major axes.

bins: int,float,list,array,string,optional
The bins for the histograms.

log: bool,optional
Whether to plot the y data on a log scale.

plot: bool,optional
Whether to plot the data or just return it.

read_in_result: bool,optional
Whether to perform system assignment again or just read it in.

filtered: bool,optional
Whether to include the filter of averaging the last 10 snapshots and removing all of the companions lesser than 0.1q.

filter_snaps_no: int,float,optional
The number of snaps to average over in the filter

min_q:int,float,optional
The mass ratio to remove companions under with the filter.

Returns
-------
x_vals: list
The bins.

weights:list
The weights of each bin

NOTE: See Plots documentation for a better description.

Example
-------
One_Snap_Plots('Mass Ratio',M2e4_C_M_J_2e7_systems[-1],M2e4_C_M_J_2e7)
    '''
    property_dist = primary_total_ratio_axis(systems,lower_limit=lower_limit,upper_limit=upper_limit,all_companions=all_companions,attribute=which_plot)
    if which_plot == 'Mass Ratio':
         x_vals,y_vals = hist(property_dist,bins = bins)
    elif which_plot == 'Semi Major Axis':
        x_vals,y_vals = hist(np.log10(property_dist)-np.log10(m_to_AU),bins = bins)
    else:
         x_vals,y_vals = hist(np.log10(property_dist),bins = bins)
    y_vals = np.insert(y_vals,0,0)
    #Creating the filtered systems
    if filtered is True:
        if snapshot is None:
            print('Please Provide Snapshot No')
            return
        if Master_File is None:
            print('Please Provide Master_File')
            return
        property_dist_filt = []
        for i in range(snapshot+1-filter_snaps_no,snapshot+1):
            property_dist_filt.append(primary_total_ratio_axis(q_filter_one_snap(Master_File[i],min_q = min_q),lower_limit=lower_limit,upper_limit=upper_limit,all_companions=all_companions,attribute=which_plot))
        x_vals_all = []
        y_vals_all = []
        count = 0
        for i in range(snapshot+1-filter_snaps_no,snapshot+1):
            if which_plot == 'Mass Ratio':
                x_vals_all.append(hist(property_dist_filt[count],bins = bins)[0])
                the_y = (hist(property_dist_filt[count],bins = bins)[1])
            elif which_plot == 'Semi Major Axis':
                x_vals_all.append(hist(np.log10(property_dist_filt[count])-np.log10(m_to_AU),bins = bins)[0])
                the_y = (hist(np.log10(property_dist_filt[count])-np.log10(m_to_AU),bins = bins)[1])
            else:
                x_vals_all.append(hist(np.log10(property_dist_filt[count]),bins = bins)[0])
                the_y = (hist(np.log10(property_dist_filt[count]),bins = bins)[1])
            the_y = np.insert(the_y,0,0)
            y_vals_all.append(the_y)
            count = count+1
        x_vals_filt = np.zeros_like(x_vals_all[-1])
        y_vals_filt = np.zeros_like(y_vals_all[-1])
        count = 0
        for i in range(snapshot+1-filter_snaps_no,snapshot+1):
            for j in range(len(x_vals)):
                x_vals_filt[j] += x_vals_all[count][j]
                y_vals_filt[j] += y_vals_all[count][j]
            count += 1
        x_vals_filt = x_vals_filt/filter_snaps_no
        y_vals_filt = y_vals_filt/filter_snaps_no
    if which_plot == 'System Mass' or which_plot == 'Primary Mass':
        if plot == True:
            #plt.title('Total Mass Distribution of all of the systems in Snapshot '+str(snapshot_number))
            if which_plot == 'System Mass':
                plt.xlabel('Log System Mass [$M_\odot$]')
            else:
                plt.xlabel('Log Primary Mass [$M_\odot$]')
            plt.ylabel('Number of Systems')
            if compare == True: #If we want to compare the total mass function to the system mass function
                if snapshot is None:
                    print('please provide snapshot')
                    return
                tot_m,vals = hist(np.log10(file[snapshot].m),bins = bins)
                vals = np.insert(vals,0,0)
                vals = vals*sum(y_vals)/sum(vals)
                plt.xlabel('Log Mass[$M_\odot$]')
                if which_plot == 'System Mass':
                    plt.step(x_vals,y_vals,label = 'Mass Dist for Systems')
                else:
                    plt.step(x_vals,y_vals,label = 'Mass Dist for Primaries')
                plt.step(tot_m,vals,label = 'Stellar Mass Dist (IMF)')
                plt.legend()
            elif filtered == True:
                plt.step(x_vals,y_vals,label = 'Raw Data')
                plt.step(x_vals_filt,y_vals_filt-0.1,label = 'After Corrections',linestyle = ':')
                plt.legend()
            else:
                plt.step(x_vals,y_vals)
            if log == True:
                plt.yscale('log')
            if filename is not None:
                plt.text(0.7,0.7,filename,transform = plt.gca().transAxes,horizontalalignment = 'left')
            plt.text(0.7,0.3,'Total Number of Systems ='+str(sum(y_vals)),transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            plt.show()
        else:
            if filtered == True:
                return x_vals_filt,y_vals_filt
            else:
                return x_vals,y_vals
    if which_plot == 'Mass Ratio':
        if plot == True:
            plt.step(x_vals,y_vals,label = 'Raw Data')
            if filtered == True:
                plt.step(x_vals_filt,y_vals_filt-0.1,label = 'After Corrections',linestyle = ':')
            plt.ylabel('Number of Systems')
            plt.xlabel('q (Companion Mass Dist)')
            if filename is not None:
                plt.text(0.5,0.7,filename,transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')  
            plt.text(0.5,0.5,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ ' $M_\odot$',transform = plt.gca().transAxes,horizontalalignment = 'left')
            if compare == True:
                if snapshot is None:
                    print('Please provide snapshots')
                    return
                if target_mass is None:
                    print('Please provide target_mass')
                    return
                Weighted_IMF,IMF = randomly_distributed_companions(systems,file,snapshot,mass_ratio=bins,plot = False,upper_limit=upper_limit,lower_limit=lower_limit,target_mass=target_mass)
                IMF = np.insert(IMF,0,0)
                plt.vlines((x_vals[-1]+x_vals[-2])/2,y_vals[-1]-np.sqrt(y_vals[-1]),y_vals[-1]+np.sqrt(y_vals[-1]),alpha = 0.3)
                plt.vlines((x_vals[4]+x_vals[3])/2,y_vals[4]-np.sqrt(y_vals[4]),y_vals[4]+np.sqrt(y_vals[4]),alpha = 0.3)
                plt.vlines((x_vals[1]+x_vals[2])/2,y_vals[2]-np.sqrt(y_vals[2]),y_vals[2]+np.sqrt(y_vals[2]),alpha = 0.3)
                if filtered is True:
                    plt.vlines((x_vals_filt[-1]+x_vals_filt[-2]+0.02)/2,y_vals_filt[-1]-np.sqrt(y_vals_filt[-1]),y_vals_filt[-1]+np.sqrt(y_vals_filt[-1]),linestyles=':')
                    plt.vlines((x_vals_filt[4]+x_vals_filt[3]+0.02)/2,y_vals_filt[4]-np.sqrt(y_vals_filt[4]),y_vals_filt[4]+np.sqrt(y_vals_filt[4]),linestyles=':')
                    plt.vlines((x_vals_filt[1]+x_vals_filt[2]+0.02)/2,y_vals_filt[2]-np.sqrt(y_vals_filt[2]),y_vals_filt[2]+np.sqrt(y_vals_filt[2]),linestyles=':')
                plt.step(x_vals,IMF*sum(y_vals)/sum(IMF),label = 'Stellar Mass Distribution (IMF)')
                if all_companions == True:
                    plt.ylabel('Number of Companions')
                else:
                    plt.step(x_vals,Weighted_IMF*sum(y_vals)/sum(Weighted_IMF),label = 'Weighted IMF')
            plt.legend()
            if log == True:
                plt.yscale('log')
        else:
            if filtered == True:
                return x_vals_filt,y_vals_filt
            else:
                return x_vals,y_vals
    if which_plot == 'Semi Major Axis':
        if plot == True:
            fig = plt.figure(figsize = (10,10))
            ax1 = fig.add_subplot(111)
            ax1.step(x_vals,y_vals,label = 'Raw Data')
            if filtered is True:
                ax1.step(x_vals_filt,y_vals_filt-0.1,label = 'After Corrections',linestyle = ':')
            ax1.vlines(np.log10(20),0,max(y_vals))
            pands = []
            for i in systems:
                if i.no>1 and lower_limit<=i.primary<=upper_limit:
                    pands.append(i.primary+i.secondary)
            average_pands = np.average(pands)*1.9891e30 
            ax1.set_xlabel('Log Semi Major Axis[AU]')
            ax2 = ax1.twiny()
            ax2.set_xlabel('Log Period[Days]')
            ax2.set_xlim(ax1.get_xlim())
            plt.ylabel('Number of Systems')
            ax1Xs = ax1.get_xticks()
            ax2Xs = []
            for X in ax1Xs:
                k = 2*np.pi*np.sqrt(((10**X*m_to_AU)**3)/(6.67e-11*average_pands))
                period = np.log10(k/(60*60*24))
                ax2Xs.append(period.round(1))
            ax2.set_xticks(ax1Xs)
            ax2.set_xbound(ax1.get_xbound())
            ax2.set_xticklabels(ax2Xs)
            if upper_limit == 1.2 and lower_limit == 0.8:
                periods = np.linspace(3.5,7.5,num = 5)
                k = ((10**periods)*24*60*60)
                smaxes3 = ((6.67e-11*(k**2)*average_pands)/(4*np.pi**2))
                smaxes = np.log10((smaxes3**(1/3))/m_to_AU)
                error_values_small = np.array([6,7,9,9,10])
                error_values_big = np.array([18,27,31,23,21])
                error_values_comb = (error_values_small+error_values_big)
                dy_comb = np.sqrt(error_values_comb)
                ax1.errorbar(smaxes,np.array(error_values_comb)*max(y_vals)/max(error_values_comb),yerr=dy_comb*max(y_vals)/max(error_values_comb),xerr = (2/3)*0.5*np.ones_like(len(smaxes)),marker = 'o',capsize = 5,color = 'black',label = 'Moe & Di Stefano 2017',linestyle = '')
            if log == True:
                plt.yscale('log')
            ax1.set_ylabel('Number of Systems')
            if all_companions == True:
                ax1.set_ylabel('Number of Sub Systems')
            ax1.legend(fontsize = 20)
            adjust_font(fig=plt.gcf(), ax_fontsize=24, labelfontsize=24)
            fig.text(0.5,0.5,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ ' $M_\odot$',transform = plt.gca().transAxes,horizontalalignment = 'left')  
            if filename is not None:
                fig.text(0.5,0.7,str(filename),transform = plt.gca().transAxes,horizontalalignment = 'left') 
        else:
            if filtered == True:
                return x_vals_filt,y_vals_filt
            else:
                return x_vals,y_vals
        
    if which_plot == 'Semi-Major Axis vs q':
        q = primary_total_ratio_axis(systems,lower_limit=lower_limit,upper_limit = upper_limit,attribute='Mass Ratio')
        smaxes = primary_total_ratio_axis(systems,lower_limit=lower_limit,upper_limit = upper_limit,attribute='Semi Major Axis')
        plt.figure(figsize= (10,10))
        #plt.title('Mass Ratio vs Semi Major Axis for a target mass of '+str(target_mass)+' in '+Files_key[systems_key])
        plt.xlabel('Semi Major Axis (in log AU)')
        plt.ylabel('Mass Ratio')
        plt.scatter(np.log10(smaxes)-np.log10(m_to_AU),q)
        if filename is not None:
            plt.text(0.7,0.7,filename,transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.show()


def Multiplicity_One_Snap_Plots(systems,Master_File = None,snapshot = None,filename = None,plot = True,multiplicity = 'Fraction',mass_break=2,bins = 'observer',filtered = False,filter_q = 0.1,filter_snaps_no =10):
    '''
Create a plot for the multiplicity over a mass range for a single snapshot.

Inputs
----------

systems: list of star system objects
The systems you want to analyze (1 snapshot or a filtered snapshot).

Parameters
----------
Master_File: list of lists of star system objects
The entire simulation with system assignment. Only required for filter on.

snapshot: int,float
The snapshot to look at. It is required with the filter on.

filename: string,optional
The name of the file to look at. It will be put on the plot if provided.

plot: bool,optional
Whether to plot or just return the values

multiplicity: bool,optional
Whether to plot for the multiplicity properties, multiplicity fraction or multiplicity frequency.

mass_break:
The spacing between masses in log space. This is used for the continous bins.

bins: int,float,list,array,string,optional
The bins for the histograms. Use continous or observer.

filtered: bool,optional
Whether to include the filter of averaging the last 10 snapshots and removing all of the companions lesser than 0.1q.

filter_snaps_no: int,float,optional
The number of snaps to average over in the filter

min_q:int,float,optional
The mass ratio to remove companions under with the filter.

Returns
-------
x_vals: list
The bins.

weights:list
The weights of each bins

NOTE: Refer to Plots documentation for a better explanation

Returns
-------
logmasslist: The masses in log space.

o1: The first output. It is the multiplicity fraction, multiplicity frequency or the primary star fraction (depending on the attribute).

o2: The second output. It is the number of multistar systems, number of companions or the single star fraction (depending on the attribute).

o3: The third output. It is the number of all systems or the companion star fraction (depending on the attribute).

NOTE: If filter is on, the filtered output will be returned.

Examples
-------
1) Multiplicity_One_Snap_Plots(M2e4_C_M_J_2e7_systems[-1],multiplicity = 'Fraction',bins = 'observer')
Simple multiplicity fraction plot.

2) Multiplicity_One_Snap_Plots(M2e4_C_M_J_2e7_systems[-1],multiplicity = 'Frequency',bins = 'observer',filtered = True,snapshot = -1,Master_File = M2e4_C_M_J_2e7_systems)
Multiplicity Frequency Plot with filter on. 

3) Multiplicity_One_Snap_Plots(M2e4_C_M_J_2e7_systems[-1],multiplicity = 'Properties',bins = 'observer',plot = False)
Multiplicity properties values being returned.
'''
    if bins is not 'observer' and bins is not 'continous':
        print('Please use the string "observer" or "continous" as the bins')
    if multiplicity == 'Frequency':
        logmasslist,o1,o2,o3 = multiplicity_frequency(systems,mass_break=mass_break,bins = bins)
    else:
        logmasslist,o1,o2,o3 = multiplicity_fraction(systems,attribute=multiplicity,mass_break=mass_break,bins = bins)
    if filtered is True:
        if snapshot is None:
            print('Please provide snapshot')
            return
        if Master_File is None:
            print('Please provide Master_File')
            return
        logmasslist_all = []
        o1_all = []
        o2_all = []
        o3_all = []
        count = 0
        for i in range(snapshot+1-filter_snaps_no,snapshot+1):
            filtered_q = q_filter_one_snap(Master_File[i],filter_q)
            if multiplicity == 'Frequency':
                logmasslist_all.append(multiplicity_frequency(filtered_q,mass_break=mass_break,bins = bins)[0])
                o1_all.append(multiplicity_frequency(filtered_q,mass_break=mass_break,bins = bins)[1])
                o2_all.append(multiplicity_frequency(filtered_q,mass_break=mass_break,bins = bins)[2])
                o3_all.append(multiplicity_frequency(filtered_q,mass_break=mass_break,bins = bins)[3])
            else:
                logmasslist_all.append(multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)[0])
                o1_all.append(multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)[1])
                o2_all.append(multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)[2])
                o3_all.append(multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)[3])
            count += 1
        logmasslist_filt = np.zeros_like(logmasslist_all[-1])
        o1_filt = np.zeros_like(o1_all[-1])
        o2_filt = np.zeros_like(o2_all[-1])
        o3_filt = np.zeros_like(o3_all[-1])
        count = 0
        for i in range(snapshot+1-filter_snaps_no,snapshot+1):
            for j in range(len(logmasslist_filt)):
                logmasslist_filt[j] += logmasslist_all[count][j]
                o1_filt[j] += o1_all[count][j]
                o2_filt[j] += o2_all[count][j]
                o3_filt[j] += o3_all[count][j]
            count += 1
        logmasslist_filt = logmasslist_filt/filter_snaps_no
        o1_filt = o1_filt/filter_snaps_no
        o2_filt = o2_filt/filter_snaps_no
        o3_filt = o3_filt/filter_snaps_no
    if multiplicity == 'Properties':
        if plot == True:
            plt.plot(logmasslist,o1,marker = '*',label = 'Primary Stars')
            plt.plot(logmasslist,o2,marker = 'o', label = 'Unbound Stars')
            plt.plot(logmasslist,o3,marker = '^',label = 'Non-Primary Stars')
            if filtered is True:
                plt.plot(logmasslist_filt,o1_filt,marker = '*',label = 'Primary Stars Filt',linestyle = ':')
                plt.plot(logmasslist_filt,o2_filt,marker = 'o', label = 'Unbound Stars Filt',linestyle = ':')
                plt.plot(logmasslist_filt,o3_filt,marker = '^',label = 'Non-Primary Stars Filt',linestyle = ':')
            plt.legend()
            plt.xlabel('Log Mass [$M_\odot$]')
            plt.ylabel('Fraction of All Stars')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            if filename is not None:
                plt.text(0.7,0.9,filename,transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
            plt.show()
        else:
            if filtered == True:
                return logmasslist_filt,o1_filt,o2_filt,o3_filt
            else:
                return logmasslist,o1,o2,o3
    if multiplicity == 'Fraction':
        if plot == True:
            if bins == 'continous':
                plt.plot(logmasslist,o1,marker = '^',label = 'Raw Data')
                if filtered is True:
                    plt.plot(logmasslist_filt,o1_filt,marker = '^',linestyle = ':',label = 'After Corrections')
            elif bins == 'observer':
                for i in range(len(logmasslist)-1):
                    if o3[i]>10 and o2[i]>0 and o2[i]<o3[i]:
                        plt.fill_between([logmasslist[i],logmasslist[i+1]],o1[i]+sigmabinom(o3[i],o2[i]),o1[i]-sigmabinom(o3[i],o2[i]),alpha = 0.6,color = '#ff7f0e')
                    else:
                        plt.fill_between([logmasslist[i],logmasslist[i+1]],o1[i]+Psigma(o3[i],o2[i]),o1[i]-Psigma(o3[i],o2[i]),alpha = 0.6,color = '#ff7f0e')
                if filtered is True:
                    for i in range(len(logmasslist_filt)-1):
                        if o3_filt[i]>10 and o2_filt[i]>0 and o2_filt[i]<o3_filt[i]:
                            plt.fill_between([logmasslist_filt[i],logmasslist_filt[i+1]],o1_filt[i]+sigmabinom(o3_filt[i],o2_filt[i]),o1_filt[i]-sigmabinom(o3_filt[i],o2_filt[i]),alpha = 0.3,color = '#1f77b4',hatch=r"\\")
                        else:
                            plt.fill_between([logmasslist_filt[i],logmasslist_filt[i+1]],o1_filt[i]+Psigma(o3_filt[i],o2_filt[i]),o1_filt[i]-Psigma(o3_filt[i],o2_filt[i]),alpha = 0.3,color = '#1f77b4',hatch=r"\\")
            error_values = [0.22,0.26,0.44,0.50,0.60,0.80]
            error_bins = [0.1,0.3,1.0,3.25,12,16]
            plt.xlabel('Log Mass [$M_\odot$]')
            plt.ylabel('Multiplicity Fraction')
            plt.errorbar(np.log10(error_bins)[0],error_values[0],yerr=[[0.04],[0.06]],xerr = 0.1,xuplims=True,marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[1],error_values[1],yerr=0.03,xerr = [[(np.log10(0.3)-np.log10(0.1))],[np.log10(0.5)-np.log10(0.3)]],marker = 'o',capsize = 5,color = 'black', label='Duchêne & Kraus 2013')
            plt.errorbar(np.log10(error_bins)[2],error_values[2],yerr=0.02,xerr = [[np.log10(1)-np.log10(0.7)],[np.log10(1.3)-np.log10(1)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[3],error_values[3],yerr=0.02,xerr = [[np.log10(3.25)-np.log10(1.5)],[np.log10(5)-np.log10(3.25)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[4],error_values[4],yerr=0.05,lolims=True,xerr = [[np.log10(12)-np.log10(8)],[np.log10(16)-np.log10(12)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[5],error_values[5],yerr=0.05,xerr = 0.1,xlolims=True,lolims = True,marker = 'o',capsize = 5,color = 'black')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            if filename is not None:
                plt.text(0.7,0.7,filename,transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
            handles, labels = plt.gca().get_legend_handles_labels()
            line = mpatches.Patch(label = 'Raw Data',color='#ff7f0e',alpha = 0.6)
            handles.extend([line])
            if filtered == True:
                line1 = mpatches.Patch(label = 'After Corrections',color='#1f77b4',alpha = 0.3, hatch=r"\\" )
                handles.extend([line1])
            plt.legend(handles = handles)
            plt.show()
        else:
            if filtered == True:
                return logmasslist_filt,o1_filt,o2_filt,o3_filt
            else:
                return logmasslist,o1,o2,o3    
    if multiplicity == 'Frequency':
        if plot == True:
            if bins == 'continous':
                plt.plot(logmasslist,o1,marker = 'o',label = 'Raw Data')
                if filtered is True:
                    plt.plot(logmasslist_filt,o1_filt,marker = 'o',label = 'After Corrections',linestyle = ':')
            elif bins == 'observer':
                for i in range(len(logmasslist)-1):
                    if o3[i]>10 and o2[i]>0 and o2[i]<o3[i]:
                        plt.fill_between([logmasslist[i],logmasslist[i+1]],o1[i]+sigmabinom(o3[i],o2[i]),o1[i]-sigmabinom(o3[i],o2[i]),alpha = 0.6,color = '#ff7f0e')
                    else:
                        plt.fill_between([logmasslist[i],logmasslist[i+1]],o1[i]+Lsigma(o3[i],o2[i]),o1[i]-Lsigma(o3[i],o2[i]),color = '#ff7f0e',alpha = 0.6)
                if filtered is True:
                    for i in range(len(logmasslist_filt)-1):
                        if o3_filt[i]>10 and o2_filt[i]>0 and o2_filt[i]<o3_filt[i]:
                            plt.fill_between([logmasslist_filt[i],logmasslist_filt[i+1]],o1_filt[i]+sigmabinom(o3_filt[i],o2_filt[i]),o1_filt[i]-sigmabinom(o3_filt[i],o2_filt[i]),alpha = 0.3,color = '#1f77b4',hatch=r"\\")
                        else:
                            plt.fill_between([logmasslist_filt[i],logmasslist_filt[i+1]],o1_filt[i]+Lsigma(o3_filt[i],o2_filt[i]),o1_filt[i]-Lsigma(o3_filt[i],o2_filt[i]),alpha = 0.3,color = '#1f77b4',hatch=r"\\")
            error_values = [0.50,0.84,1.3,1.6,2.1]
            error_bins = [1.0,3.5,7.0,12.5,16]
            plt.xlabel('Mass (in log solar masses)')
            plt.ylabel('Multiplicity Frequency')
            plt.errorbar(np.log10(error_bins)[0],error_values[0],yerr=0.04,xerr = [[(np.log10(1.0)-np.log10(0.8))],[np.log10(1.2)-np.log10(1.0)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[1],error_values[1],yerr=0.11,xerr = [[(np.log10(3.5)-np.log10(2))],[(np.log10(5)-np.log10(3.5))]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[2],error_values[2],yerr=0.2,xerr = [[(np.log10(7)-np.log10(5))],[(np.log10(9)-np.log10(7))]],marker = 'o',capsize = 5,color = 'black',label = 'Moe & DiStefano 2017')
            plt.errorbar(np.log10(error_bins)[3],error_values[3],yerr=0.2,xerr = [[np.log10(12.5)-np.log10(9)],[np.log10(16)-np.log10(12.5)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[4],error_values[4],yerr=0.3,xlolims=True,xerr = 0.1,marker = 'o',capsize = 5,color = 'black')
            if filename is not None:
                plt.text(0.7,0.7,filename,transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
            handles, labels = plt.gca().get_legend_handles_labels()
            line = mpatches.Patch(label = 'Raw Data',color='#ff7f0e',alpha = 0.6)
            handles.extend([line])
            if filtered == True:
                line1 = mpatches.Patch(label = 'After Corrections',color='#1f77b4',alpha = 0.3, hatch=r"\\" )
                handles.extend([line1])
            plt.legend(handles = handles)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            plt.show()
            plt.figure()
        else:
            if filtered == True:
                return logmasslist_filt,o1_filt,o2_filt,o3_filt
            else:
                return logmasslist,o1,o2,o3

def Time_Evolution_Plots(which_plot,Master_File,file,steps = 1,target_mass = 1,T = [1],dt = [0.5],target_age = 1,filename = None,min_age = 0,read_in_result = True,start = 0,upper_limit = 1.3,lower_limit = 0.7,plot = True,multiplicity = 'Fraction',zero = 'Consistent Mass',select_by_time = True):
    '''
Create a plot for a property that evolves through the simulation.

Inputs
----------
which_plot: string
The plot to be made.

Master_File: list of lists of star system objects
The entire simulation with system assignment. Only required for time evolution plots and filter on.

file:list of sinkdata objects
The original sinkdata file.

Parameters
----------
steps: int,optional
The number of snapshots per bin in multiplicity over time.

target_mass: int,float,optional
The mass of the primaries of the systems of interest.

T: int,float,optional
The time from end of the simulation to select from

dt: int,float,optional
The tolerance of T. For example, if the total runtime is 10 Myr, T = 2 and dt = 0.2, then it looks at stars formed between 7.9-8.1 Myrs.

target_age:int,float,optional
The maximum age for the YSO multiplicities.

filename:string
The name of the original file. It will be labelled on the plot if provided.

min_age:int,float,optional
The minimum age for the YSO multiplicity

read_in_result: bool,optional
Whether to perform system assignment again or just read it in.

start: int,optional
Starting point of multiplicity time evolution

upper_limit: int,float,optional
The upper mass limit of the primaries of systems of interest.

lower_limit: int,float,optional
The lower mass limit of the primaries of systems of interest.

plot: bool,optional
Whether to plot the data or just return it.

multiplicity: bool,optional
Whether to plot for the multiplicity properties(only by mass plot), multiplicity fraction or multiplicity frequency.

zero: string,optional
Whether to set the zero age as 'formation' (where the star formed) or 'consistent mass' (where the star stopped accreting)

select_by_time: bool,optional
Whether to look at average multiplicity for all stars or only those in a window.

Returns
-------
x_vals: list
The bins.

weights:list
The weights of each bin

Examples
-------
1) Time_Evolution_Plots("Multiplicity Time Evolution",M2e4_C_M_J_2e7_systems,M2e4_C_M_J_2e7,multiplicity = 'Fraction',target_mass = 1')
The multiplicity at every time for the given target mass.

2)Time_Evolution_Plots("Multiplicity Lifetime Evolution",M2e4_C_M_J_2e7_systems,M2e4_C_M_J_2e7,multiplicity = 'Fraction',target_mass = 1,T = [1,2,3],dt = [0.5,0.5,0.5]')
The multiplicity of stars of the target mass born at the given times.

3)Time_Evolution_Plots("YSO Multiplicity",M2e4_C_M_J_2e7_systems,M2e4_C_M_J_2e7,min_age = 0,target_age = 1)
The multiplicity of stars of younger than the target age and older than the minimum age.
'''
    if which_plot == 'Multiplicity Time Evolution':
        if Master_File is None:
            print('provide master file')
            return
        elif filename is None:
            print('Provide the filename')
        if multiplicity == 'Fraction':
            if plot == True:
                Multiplicity_Fraction_Time_Evolution(file,Master_File,filename,steps = steps,target_mass=target_mass,read_in_result=read_in_result,start = start,upper_limit=upper_limit,lower_limit=lower_limit,plot = True)
            elif plot == False:
                return Multiplicity_Fraction_Time_Evolution(file,Master_File,filename,steps = steps,target_mass=target_mass,read_in_result=read_in_result,start = start,upper_limit=upper_limit,lower_limit=lower_limit,plot = False)
        if multiplicity == 'Frequency':
            if plot == True:
                Multiplicity_Frequency_Time_Evolution(file,Master_File,filename,steps = steps,target_mass=target_mass,read_in_result=read_in_result,start = start,upper_limit=upper_limit,lower_limit=lower_limit,plot = True)
            elif plot == False:
                return Multiplicity_Frequency_Time_Evolution(file,Master_File,filename,steps = steps,target_mass=target_mass,read_in_result=read_in_result,start = start,upper_limit=upper_limit,lower_limit=lower_limit,plot = False)
    if which_plot == 'Multiplicity Lifetime Evolution':
        if Master_File is None:
            print('provide master file')
            return
        if plot is False:
            print('Use Plot == True')
            return
        multiplicity_and_age_combined(file,Master_File,filename = filename,T_list=T,dt_list=dt,upper_limit=upper_limit,lower_limit=lower_limit,target_mass=target_mass,zero = zero,multiplicity=multiplicity)
    if which_plot == 'YSO Multiplicity':
        if Master_File is None:
            print('provide master file')
            return
        if filename is None:
            print('Please Provide filename')
        mul1,cou1,av1 = YSO_multiplicity(file,Master_File,target_age = target_age,min_age=min_age)
        times = []
        prop_times = []
        start_snap = Mass_Creation_Finder(file,min_mass = 0)
        start_time = file[start_snap].t*time_to_Myr
        for i in range(len(file)):
            times.append(file[i].t*time_to_Myr - start_time)
        for i in range(len(file)):
            prop_times.append(file[i].t)
        end_snap = closest(prop_times,prop_times[-1]-target_age,param = 'index')
        
        prop_times = np.array(prop_times)
        ff_t = t_ff(file_properties(filename,param = 'm'),file_properties(filename,param = 'r'))
        prop_times = (prop_times/ff_t)
        
        plt.plot(prop_times,mul1,label ='< '+str(target_age)+' Myr stars in simulation')
        
        left_limit = plt.xlim()[0]
        right_limit = plt.xlim()[1]
        
        plt.fill_betweenx(np.linspace(0.35,0.5,100),left_limit,right_limit,color = 'orange',alpha = 0.3)
        plt.fill_betweenx(np.linspace(0.3,0.4,100),left_limit,right_limit,color = 'black',alpha = 0.3)
        plt.fill_betweenx(np.linspace(0.25,0.15,100),left_limit,right_limit,color = 'purple',alpha = 0.3)
        plt.text(0.1,0.45,'Class 0 Perseus',fontsize = 20)
        plt.text(0.1,0.32,'Class 0 Orion',fontsize = 20)
        plt.text(0.1,0.2,'Class 1 Orion',fontsize = 20)
        
        plt.xlabel(r'Time [$\frac{t}{t_ff}$]')
        plt.ylabel('YSO Multiplicity Fraction')
        adjust_font(fig=plt.gcf(), ax_fontsize=24, labelfontsize=24)
        plt.xlim((left_limit,right_limit))
        plt.figure()
        if filename is not None:
            plt.text(0.7,0.7,filename,transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
        plt.plot(prop_times,cou1,label = '< '+str(target_age)+' Myr Stars in Simulation')
        plt.legend()
        
        plt.yscale('log')
        plt.xlabel(r'Time [$\frac{t}{t_ff}$]')
        plt.ylabel('Number of Young Stars')
        #plt.legend()
        plt.figure()
        plt.plot(prop_times,av1,label = 'Formation')
        #[start_snap:end_snap]
        plt.yscale('log')
        #plt.plot(times,av2,label = 'Consistent Mass')
        plt.xlabel(r'Time [$\frac{t}{t_ff}$]')
        plt.ylabel('Average Mass of Young Stars')
#Function that contains all the plots

#Can split it up into smaller ones and then put those into a mega function as Plots() so it can be called as one but has subparts

def Plots(which_plot,systems,file,filename = None,Master_File = None,snapshot= None,target_mass=1,target_age=1,upper_limit = 1.3,lower_limit = 0.7,mass_break = 2,T = [1],dt = [0.5],min_age = 0,all_companions = True,bins = 10,log = True,compare = False,plot = True,multiplicity = 'Fraction',steps = 1,read_in_result = True,start = 0,zero = 'Formation',select_by_time = True,filtered = False,filter_snaps_no = 10,min_q = 0.1): 
    '''
Create a plot or gives you the values to create a plot for the whole system.

Inputs
----------
which_plot: string
The plot to be made.

systems: list of star system objects
The systems you want to analyze (1 snapshot or a filtered snapshot).

file:list of sinkdata objects
The original sinkdata file.

filename:string
The name of the original file. It will be labelled on the plot if provided.

Master_File: list of lists of star system objects
The entire simulation with system assignment. Only required for time evolution plots and filter on.

snapshot: int,float
The snapshot number you want to look at. Only required for IMF comparisons and filter on.

Parameters
----------

target_mass: int,float,optional
The mass of the primaries of the systems of interest.

target_age:int,float,optional
The maximum age for the YSO multiplicities.

upper_limit: int,float,optional
The upper mass limit of the primaries of systems of interest.

lower_limit: int,float,optional
The lower mass limit of the primaries of systems of interest.

mass_break: int,float,optional
The spacing between masses in log space (important for multiplicity fraction)

T: list,optional
The times from end of the simulation to select from. 

dt: list,optional
The tolerance of T. 

min_age:int,float,optional
The minimum age for the YSO multiplicity

all_companions: bool,optional
Whether to include all companions in the mass ratio or semi major axes.

bins: int,float,list,array,string,optional
The bins for the histograms.

log: bool,optional
Whether to plot the y data on a log scale.

compare: bool,optional
Whether to include the IMF for comparison.

plot: bool,optional
Whether to plot the data or just return it.

multiplicity: bool,optional
Whether to plot for the multiplicity properties(only by mass plot), multiplicity fraction or multiplicity frequency.

steps: int,optional
The number of snapshots per bin in multiplicity over time.

read_in_result: bool,optional
Whether to perform system assignment again or just read it in.

start: int,optional
Starting point of multiplicity time evolution

zero: string,optional
Whether to set the zero age as 'formation' (where the star formed) or 'consistent mass' (where the star stopped accreting)

select_by_time: bool,optional
Whether to look at average multiplicity for all stars or only those in a window.

filtered: bool,optional
Whether to include the filter of averaging the last 10 snapshots and removing all of the companions lesser than 0.1q.

filter_snaps_no: int,float,optional
The number of snaps to average over in the filter

min_q:int,float,optional
The mass ratio to remove companions under with the filter.

Returns
-------
x_vals: list
The bins.

weights:list
The weights of each bin
'''
    One_System_Plots = ['System Mass','Primary Mass','Semi Major Axis','Mass Ratio','Semi Major Axis vs q']
    Time_Evo_Plots = ['Multiplicity Time Evolution','Multiplicity Lifetime Evolution','YSO Multiplicity']
    if which_plot in One_System_Plots:
        if plot == True:
            One_Snap_Plots(which_plot,systems,file,filename = filename,snapshot = snapshot,upper_limit = upper_limit,lower_limit = lower_limit,target_mass = target_mass,all_companions = all_companions,bins = bins,log = log,compare = compare,plot = plot,read_in_result = read_in_result,filtered = filtered,filter_snaps_no = filter_snaps_no,min_q = min_q,Master_File=Master_File)
        else:
            return One_Snap_Plots(which_plot,systems,file,filename = filename,snapshot = snapshot,upper_limit = upper_limit,lower_limit = lower_limit,target_mass = target_mass,all_companions = all_companions,bins = bins,log = log,compare = compare,plot = plot,read_in_result = read_in_result,filtered = filtered,filter_snaps_no = filter_snaps_no,min_q = min_q,Master_File=Master_File)
    elif which_plot == 'Multiplicity':
        if plot == True:
            Multiplicity_One_Snap_Plots(systems,Master_File,multiplicity = multiplicity,mass_break=mass_break,bins = bins,filtered = filtered,filter_q = min_q,plot = plot,filename = filename,snapshot = snapshot,filter_snaps_no =filter_snaps_no)
        else:
            return Multiplicity_One_Snap_Plots(systems,Master_File,multiplicity = multiplicity,mass_break=mass_break,bins = bins,filtered = filtered,filter_q = min_q,plot = plot,filename = filename,snapshot = snapshot,filter_snaps_no =filter_snaps_no)
    elif which_plot in Time_Evo_Plots:
        if plot == True:
            Time_Evolution_Plots(which_plot,Master_File,file,filename=filename,steps = steps,target_mass = target_mass,T = T,dt = dt,target_age = target_age,min_age = min_age,read_in_result = read_in_result,start = start,upper_limit = upper_limit,lower_limit = lower_limit,plot = plot,multiplicity = multiplicity,zero = zero,select_by_time = select_by_time)
        else:
            return Time_Evolution_Plots(which_plot,Master_File,file,filename=filename,steps = steps,target_mass = target_mass,T = T,dt = dt,target_age = target_age,min_age = min_age,read_in_result = read_in_result,start = start,upper_limit = upper_limit,lower_limit = lower_limit,plot = plot,multiplicity = multiplicity,zero = zero,select_by_time = select_by_time)

def Multi_Plot(which_plot,Systems,Files,Filenames,Snapshots = None,log = False,upper_limit = 1.3,lower_limit = 0.7,target_mass = 1,target_age = 1,min_age = 0,multiplicity = 'Fraction',all_companions = True,filtered = False,normalized = True,norm_no = 100,time_plot = 'consistent mass'):
    '''
Creates distribution plots for more than one file
Inputs
----------
which_plot: string
The plot to be made.

Systems: list of list of star system objects
All of the Systems from all of the files you want to see.

Files:list of list of sinkdata objects
The list of all the files you want to see.

Filenames: list of strings
The names of the files that you want to see.

Snapshots: int,float
The snapshot number you want to look at. By default, it looks at the last one.

Parameters
----------

log: bool,optional
Whether to plot the y data on a log scale.

upper_limit: int,float,optional
The upper mass limit of the primaries of systems of interest.

lower_limit: int,float,optional
The lower mass limit of the primaries of systems of interest.

target_mass: int,float,optional
The mass of the primaries of the systems of interest.

multiplicity: bool,optional
Whether to plot for the multiplicity properties(only by mass plot), multiplicity fraction or multiplicity frequency.

all_companions: bool,optional
Whether to include all companions in the mass ratio or semi major axes.

filtered: bool,optional:
Whether to include the filtered results or the unfiltered results

normalized:bool,optional:
Whether to normalize the systems to a certain number

norm_no: int,optional:
The number of systems to normalize to.

time_plot:str,optional: = 'consistent mass'
Whether to plot the consistent mass or all of the stars in the multiplicity time evolution.
Examples
----------
1) Multi_Plot('Mass Ratio',Systems,Files,Filenames,normalized=True)


'''  
    if Snapshots == None:
        Snapshots = [[-1]]*len(Filenames)
    Snapshots = list(flatten(Snapshots))
    x = []
    y = []
    if which_plot == 'System Mass':
        bins = np.linspace(-1,3,8)
        plt.xlabel('Log System Mass [$M_\odot$]')
        plt.ylabel('Number of Systems')
    if which_plot == 'Primary Mass':
        bins = np.linspace(-1,3,8)
        plt.xlabel('Log Primary Mass [$M_\odot$]')
        plt.ylabel('Number of Systems')
    if which_plot == 'Mass Ratio':
        bins = np.linspace(0,1,11)
        plt.xlabel('q (Companion Mass Ratio)')
        plt.ylabel('Number of Systems')
        if all_companions is True:
            plt.ylabel('Number of Companions')
    if which_plot == 'Multiplicity':
        bins = 'observer'
        plt.xlabel('Log Mass [$M_\odot$]')
        if multiplicity == 'Fraction':
            plt.ylabel('Multiplicity Fraction')
        if multiplicity == 'Frequency':
            plt.ylabel('Multiplicity Frequency')
    if which_plot == 'Semi Major Axis':
        bins = np.linspace(-1,7,13)
    if which_plot == 'Multiplicity':
        error = []
    times = []
    fractions = []
    cons_fracs = []
    nos = []
    avg_mass = []
    for i in tqdm(range(0,len(Filenames)),desc = 'Getting Data',position=0):
        if which_plot == 'Multiplicity':
            a,b,c,d = Plots(which_plot,Systems[i][Snapshots[i]],Files[i],log = False,plot = False,bins = bins,upper_limit = upper_limit,lower_limit = lower_limit,multiplicity = multiplicity,all_companions = all_companions,filtered = filtered,snapshot = Snapshots[i],Master_File = Systems[i])
            comp_mul_no = c
            sys_no = d
            error_one = []
            for i in range(len(sys_no)):
                if sys_no[i]>10 and comp_mul_no[i]>0 and comp_mul_no[i]<sys_no[i]:
                    error_one.append(sigmabinom(sys_no[i],comp_mul_no[i]))
                else:
                    if multiplicity == 'Fraction':
                        error_one.append(Psigma(sys_no[i],comp_mul_no[i]))
                    elif multiplicity == 'Frequency':
                        error_one.append(Lsigma(sys_no[i],comp_mul_no[i]))
            error.append(error_one)
            x.append(a)
            y.append(b)
        elif which_plot == 'Multiplicity Time Evolution':
            if multiplicity == 'Fraction':
                time,fraction,cons_frac = Multiplicity_Fraction_Time_Evolution(Files[i],Systems[i],Filenames[i],upper_limit=upper_limit,lower_limit=lower_limit,plot = False)
                times.append(time)
                fractions.append(fraction)
                cons_fracs.append(cons_frac)
            elif multiplicity == 'Frequency':
                time,fraction,cons_frac = Multiplicity_Frequency_Time_Evolution(Files[i],Systems[i],Filenames[i],upper_limit=upper_limit,lower_limit=lower_limit,plot = False)
                times.append(time)
                fractions.append(fraction)
                cons_fracs.append(cons_frac)
        elif which_plot == 'YSO Multiplicity':
            time = []
            for j in range(len(Files[i])):
                time.append(Files[i][j].t)
            time = np.array(time)
            ff_t = t_ff(file_properties(Filenames[i],param = 'm'),file_properties(Filenames[i],param = 'r'))
            time = (time/(ff_t*np.sqrt(file_properties(Filenames[i],param = 'alpha'))))
            times.append(time)
            fraction,no,am = YSO_multiplicity(Files[i],Systems[i],min_age = min_age,target_age = target_age)
            fractions.append(fraction)
            nos.append(no)
            avg_mass.append(am)
        else:
            a,b = Plots(which_plot,Systems[i][Snapshots[i]],Files[i],log = False,plot = False,bins = bins,upper_limit = upper_limit,lower_limit = lower_limit,multiplicity = multiplicity,all_companions = all_companions,filtered = filtered,snapshot = Snapshots[i],Master_File = Systems[i])
            if normalized == True:
                b = b*norm_no/sum(b)
            x.append(a)
            y.append(b)
    if which_plot == 'Semi Major Axis':
        fig = plt.figure(figsize = (10,10))
        ax1 = fig.add_subplot(111)
        for i in range(len(Files)):
            ax1.step(x[i],y[i],label = Filenames[i])
        ax1.vlines(np.log10(20),0,max(y[0]))
        pands = []
        for i in Systems[0][-1]:
            if i.no>1 and lower_limit<=i.primary<=upper_limit:
                pands.append(i.primary+i.secondary)
        average_pands = np.average(pands)*1.9891e30 
        ax1.set_xlabel('Log Semi Major Axis[AU]')
        ax2 = ax1.twiny()
        ax2.set_xlabel('Log Period[Days]')
        ax2.set_xlim(ax1.get_xlim())
        ax1.set_ylabel('Number of Systems')
        if all_companions == True:
            ax1.set_ylabel('Number of Sub-Systems')
        ax1Xs = ax1.get_xticks()
        ax2Xs = []
        for X in ax1Xs:
            k = 2*np.pi*np.sqrt(((10**X*m_to_AU)**3)/(6.67e-11*average_pands))
            period = np.log10(k/(60*60*24))
            ax2Xs.append(period.round(1))
        ax2.set_xticks(ax1Xs)
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(ax2Xs)
        if upper_limit == 1.3 and lower_limit == 0.7:
            periods = np.linspace(3.5,7.5,num = 5)
            k = ((10**periods)*24*60*60)
            smaxes3 = ((6.67e-11*(k**2)*average_pands)/(4*np.pi**2))
            smaxes = np.log10((smaxes3**(1/3))/m_to_AU)
            error_values_small = np.array([6,7,9,9,10])
            error_values_big = np.array([18,27,31,23,21])
            error_values_comb = (error_values_small+error_values_big)
            dy_comb = np.sqrt(error_values_comb)
            ax1.errorbar(smaxes,np.array(error_values_comb)*max(y[0])/max(error_values_comb),yerr=dy_comb*max(y[0])/max(error_values_comb),xerr = (2/3)*0.5*np.ones_like(len(smaxes)),marker = 'o',capsize = 5,color = 'black',label = 'Moe & Di Stefano 2017',linestyle = '')
        ax1.legend()
    elif which_plot == 'Multiplicity':
        for i in range(0,len(Filenames)):
            plt.plot(x[i],y[i],label = Filenames[i])
            plt.fill_between(x[i],np.array(y[i],dtype = np.float32)+error[i],np.array(y[i],dtype = np.float32)-error[i],alpha = 0.15)
        if multiplicity == 'Fraction':
            error_values = [0.22,0.26,0.44,0.50,0.60,0.80]
            error_bins = [0.1,0.3,1.0,3.25,12,16]
            plt.errorbar(np.log10(error_bins)[0],error_values[0],yerr=[[0.04],[0.06]],xerr = 0.1,xuplims=True,marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[1],error_values[1],yerr=0.03,xerr = [[(np.log10(0.3)-np.log10(0.1))],[np.log10(0.5)-np.log10(0.3)]],marker = 'o',capsize = 5,color = 'black', label='Duchêne & Kraus 2013')
            plt.errorbar(np.log10(error_bins)[2],error_values[2],yerr=0.02,xerr = [[np.log10(1)-np.log10(0.7)],[np.log10(1.3)-np.log10(1)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[3],error_values[3],yerr=0.02,lolims = True,xerr = [[np.log10(3.25)-np.log10(1.5)],[np.log10(5)-np.log10(3.25)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[4],error_values[4],yerr=0.05,lolims=True,xerr = [[np.log10(12)-np.log10(8)],[np.log10(16)-np.log10(12)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[5],error_values[5],yerr=0.05,xerr = 0.1,xlolims=True,lolims = True,marker = 'o',capsize = 5,color = 'black')
            plt.ylim([-0.01,1.01])
        elif multiplicity == 'Frequency':
            error_values = [0.50,0.84,1.3,1.6,2.1]
            error_bins = [1.0,3.5,7.0,12.5,16]
            plt.errorbar(np.log10(error_bins)[0],error_values[0],yerr=0.04,xerr = [[(np.log10(1.0)-np.log10(0.8))],[np.log10(1.2)-np.log10(1.0)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[1],error_values[1],yerr=0.11,xerr = [[(np.log10(3.5)-np.log10(2))],[(np.log10(5)-np.log10(3.5))]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[2],error_values[2],yerr=0.2,xerr = [[(np.log10(7)-np.log10(5))],[(np.log10(9)-np.log10(7))]],marker = 'o',capsize = 5,color = 'black',label = 'Moe & DiStefano 2017')
            plt.errorbar(np.log10(error_bins)[3],error_values[3],yerr=0.2,xerr = [[np.log10(12.5)-np.log10(9)],[np.log10(16)-np.log10(12.5)]],marker = 'o',capsize = 5,color = 'black')
            plt.errorbar(np.log10(error_bins)[4],error_values[4],yerr=0.3,xlolims=True,xerr = 0.1,marker = 'o',capsize = 5,color = 'black')
            plt.ylim([-0.01,3.01])
        plt.legend()
    elif which_plot == 'Multiplicity Time Evolution' or which_plot == 'YSO Multiplicity':
        for i in range(len(Files)):
            if time_plot == 'consistent mass' and which_plot == 'Multiplicity Time Evolution':
                plt.plot(times[i],cons_fracs[i],label = Filenames[i])
            elif (time_plot == 'all' and which_plot == 'Multiplicity Time Evolution') or which_plot == 'YSO Multiplicity':
                plt.plot(times[i],fractions[i],label = Filenames[i])
        plt.xlabel(r'Time [$\frac{t}{t_{ff}}$]')
        if multiplicity == 'Fraction':
            plt.ylabel('Multiplicity Fraction')
        if multiplicity == 'Frequency':
            plt.ylabel('Multiplicity Frequency')
        elif which_plot == 'YSO Multiplicity':
            plt.ylabel('YSO Multiplicity Fraction')
            plt.fill_betweenx(np.linspace(0.35,0.5,100),0,max(list(flatten(times))),color = 'orange',alpha = 0.3)
            plt.fill_betweenx(np.linspace(0.3,0.4,100),0,max(list(flatten(times))),color = 'black',alpha = 0.3)
            plt.fill_betweenx(np.linspace(0.25,0.15,100),0,max(list(flatten(times))),color = 'purple',alpha = 0.3)
            plt.text(0.1,0.45,'Class 0 Perseus',fontsize = 20)
            plt.text(0.1,0.32,'Class 0 Orion',fontsize = 20)
            plt.text(0.1,0.2,'Class 1 Orion',fontsize = 20)
            plt.legend()
            plt.figure()
            for i in range(len(Files)):
                plt.plot(times[i],nos[i],label = Filenames[i])
            plt.ylabel('Number of YSOs')
            plt.xlabel(r'Time [$\frac{t}{t_{ff}}$]')
            plt.legend()
            plt.figure()
            for i in range(len(Files)):
                plt.plot(times[i],avg_mass[i],label = Filenames[i])
            plt.ylabel('Average Mass of YSOs')
            plt.xlabel(r'Time [$\frac{t}{t_{ff}}$]')
            plt.legend()
        plt.legend()
        plt.legend()
    else:
        for i in range(0,len(Filenames)):
            plt.step(x[i],y[i],label = Filenames[i])
        plt.legend()
    adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
    if log == True:
        plt.yscale('log')

#The length for the given box plot
L = (4/3*np.pi)**(1/3)*10

