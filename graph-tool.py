import numpy as np
from sklearn.neighbors import KDTree 
import pickle # to save/load kdtree
import pandas as pd
from graph_tool.all import *
from configHelper import *

# set np print option to avoid scientific notation
# np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# read raw data from binary file
def readData():
	# read config for input data
	binaryFile = getConfig("Files","depositBin")
	# import raw data to np
	return np.fromfile(binaryFile, dtype=np.float16)

def normalize(data):
	# data = data[data>0]
	return ((data - np.min(data)) / (np.max(data) - np.min(data)))

# get non zero (3D) indexes from raw data
def getNonZeroData():
	data = readData()
	# print(rawData.shape)

	print("len(data) b4 preprocess: {}".format(len(data)))

	# reshape and convert data to suitable formats
	brickSize  = int(getConfig("PreProcess","brickSize"))
	voxels_f16 = data.reshape((brickSize, brickSize, brickSize))

	# normalize data
	if isOperationSet(section="PreProcess",operation="normalize"):
		voxels_f16 = normalize(voxels_f16)

	densityThreshold = float(getConfig("PreProcess","densityThreshold"))
	x,y,z = np.where(voxels_f16 >= densityThreshold)

	# values (>= densityThreshold) of the array
	v = voxels_f16[x,y,z]
	
	return np.vstack((x, y, z, v)).T

def applyThresholdOnData(data):
	# remove particles below defined threshold
	densityThreshold = float(getConfig("PreProcess","densityThreshold"))

	# normalize data
	if isOperationSet(section="PreProcess",operation="normalize"):
		data = normalize(data)

	data = data[data>=densityThreshold]

	print("len(data) preproced: {}".format(len(data)))

	return data

def getCentralSlice(nSubSample=30,axis=2):
	data = readData()
	# print(rawData.shape)

	print("len(data) b4 preprocess: {}".format(len(data)))

	# reshape data to get whole original cube
	brickSize  = int(getConfig("PreProcess","brickSize"))
	voxels_f16 = data.reshape((brickSize, brickSize, brickSize))

	startCenter = int(((nSubSample//2)*brickSize)/nSubSample)
	endCenter	= int((((nSubSample//2)+1)*brickSize)/nSubSample)

	subCube = voxels_f16[startCenter:endCenter, startCenter:endCenter, startCenter:endCenter]

	centralSlice = subCube[:,:,int((brickSize/nSubSample)/2)]
	
	print(centralSlice.shape)

	### here, centralSlice is the "raw data"
	### that we can compare/plot with graph 
	
	densityThreshold = float(getConfig("PreProcess","densityThreshold"))
	x,y = np.where(centralSlice >= densityThreshold)

	# values (>= densityThreshold) of the array
	v = centralSlice[x,y]
	
	return np.vstack((x, y, v)).T	

def getCenterSubSample():
	# thinking on 27 subcubes to get the central one

	data = readData()
	# print(rawData.shape)

	print("len(data) b4 preprocess: {}".format(len(data)))

	# reshape data to get whole original cube
	brickSize	= int(getConfig("PreProcess","brickSize"))
	nSubSample	= int(getConfig("Graph","nSubSample"))
	centerShift = int(getConfig("Graph","centerShift"))
	voxels_f16	= data.reshape((brickSize, brickSize, brickSize))
	print(voxels_f16.shape)

	# general form
	startCenter = int(((nSubSample//2)*brickSize)/nSubSample)+centerShift
	endCenter	= int((((nSubSample//2)+1)*brickSize)/nSubSample)+centerShift

	print(startCenter,endCenter)

	subCube = voxels_f16[startCenter:endCenter, startCenter:endCenter, startCenter:endCenter]

	print(subCube.shape)

	# normalize data
	if isOperationSet(section="PreProcess",operation="normalize"):
		subCube = normalize(subCube)
	
	densityThreshold = float(getConfig("PreProcess","densityThreshold"))
	x,y,z = np.where(subCube >= densityThreshold)

	# values (>= densityThreshold) of the array
	v = subCube[x,y,z]
	
	return np.vstack((x, y, z, v)).T


def getSplitData():
	rawData = readData()
	
	# for 8 subCubes from 720^3 cube
	subCubes = rawData.reshape(2,360,2,360,2,360).transpose(0,2,4,1,3,5).reshape(-1,360,360,360)

	# for 27 subcubes from 360^3 cube
	subCubes = rawData.reshape(3,120,3,120,3,120).transpose(0,2,4,1,3,5).reshape(-1,120,120,120)

	# later...?
	# for subCube in subCubes:

def getData():
	data = getCenterSubSample()
	# data = []
	# if subCube:
		# data = getCenterSubSample()
	# else: 
		# data = getNonZeroData()

	print("len(data) preprocessed: {}".format(len(data)))

	return data

def getData_2D():
	data = getCentralSlice()

	print("len(data) preprocessed: {}".format(len(data)))

	return data


# create data files
def createDataFiles():
	print("createDataFiles():")
	# get positions from non zero particles
	# data = applyThresholdOnData(getNonZeroData())
	
	# CHANGE HERE FOR 2D OR 3D
	data = getData()

	# save csv
	depositCSV = getConfig("Files","depositCSV")
	np.savetxt(depositCSV, data, delimiter = ',', header="x,y,z,v")

	# save kdtree
	kdtree = KDTree(data)
	kdtreeFile = getConfig("Files","kdtree")
	# dumping kdtree to bin file
	with open(kdtreeFile, 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
		pickle.dump(kdtree, f, pickle.HIGHEST_PROTOCOL)

# create data files
def createDataFiles_2D():
	print("createDataFiles():")
	# get positions from non zero particles
	# data = applyThresholdOnData(getNonZeroData())
	data = getData_2D()

	# save csv
	depositCSV = getConfig("Files","depositCSV")
	np.savetxt(depositCSV, data, delimiter = ',', header="x,y,v")

	# save kdtree
	kdtree = KDTree(data)
	kdtreeFile = getConfig("Files","kdtree")
	# dumping kdtree to bin file
	with open(kdtreeFile, 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
		pickle.dump(kdtree, f, pickle.HIGHEST_PROTOCOL)

def getPandasDF():
	# load csv with position and density --> vertices
	print("loading positions csv.\n")
	depositCSV  = getConfig("Files","depositCSV")
	f = open(depositCSV, 'r')
	header = f.readline().replace('# ', '').replace('\n', '').split(',')
	# read csv specifying dtypes (avoid memory issues)
	return pd.read_csv(depositCSV, skiprows=1, names=header, dtype={'x': np.int32, 'y': np.int32, 'z': np.int32, 'v': np.float64})

def getPandasDF_2D():
	# load csv with position and density --> vertices
	print("loading positions csv.\n")
	depositCSV  = getConfig("Files","depositCSV")
	f = open(depositCSV, 'r')
	header = f.readline().replace('# ', '').replace('\n', '').split(',')
	# read csv specifying dtypes (avoid memory issues)
	return pd.read_csv(depositCSV, skiprows=1, names=header, dtype={'x': np.int32, 'y': np.int32, 'v': np.float64})

def createGraph():
	print("createGraph():")
	# # FOR NOW, WE DON'T NEED THE CSV POSITIONS (right?!)	
	# df_vertices = getPandasDF()

	# loading the data needed (?) for tree queries & graph 
	# data = applyThresholdOnData(getNonZeroData())

	# CHANGE HERE FOR 2D OR 3D
	data = getData()
	
	# load kdtree with distances --> edges
	print("loading kdtree.")
	kdtreeFile = getConfig("Files","kdtree")
	with open(kdtreeFile, 'rb') as f:
		kdtree = pickle.load(f)

	# we create a graph with
	## * N vertices (without properties)
	print("creating graph...")
	base_graph = Graph(directed=False)
	# vlist has indexes for df_vertices
	# vlist = list(base_graph.add_vertex(len(df_vertices)))
	len_vert = len(kdtree.data)
	base_graph.add_vertex(len_vert)
	print("with {} vertices.".format(len_vert))

	maxDistEdge = float(getConfig("Graph","maxDistEdge"))

	print("maxDistEdge = {}".format(maxDistEdge))
	print("querying kdtree...")
	# nearest_ind_r is an array of arrays with indexes only
	nearest_ind_r = kdtree.query_radius(data, r=maxDistEdge, return_distance=False)
	
	# remember to "jump 0" to avoid identity (no loop edge)
	
	# edge_list = np.array([])
	edge_list = []
	print("starting edge loop...")

	# out of bounds error only for 720_r2 (must be memory)
	# IndexError: index 2802977 is out of bounds for axis 0 with size 2802977

	# for u_ind in range(len_vert-1):
	for u_ind in range(len_vert):
		for v_ind in nearest_ind_r[u_ind][1:]:
			new_edge = (u_ind, v_ind)
			edge_list.append(new_edge)
			# np.append(edge_list, new_edge)

	edge_list = np.array(edge_list)
	print("edge loop done.\n")

	base_graph.add_edge_list(edge_list)
	print("{} edges added to graph.\n".format(len(edge_list)))

	# save graph
	outputPath = getConfig("Paths","outputPath")
	baseGraphFile = getConfig("Files","baseGraph")
	print("saving graph (file prefix: {}{}).\n".format(outputPath, baseGraphFile))
	base_graph.save(outputPath+baseGraphFile+".gt", fmt="gt")
	base_graph.save(outputPath+baseGraphFile+".xml", fmt="xml")

# assign edge to isolated vertices
# if user wants to ensure connectivity
def connectVertices(graph=""):
	if graph == "":
		baseGraphFile = getConfig("Graph","base_graph")
		graph = base_graph.load(baseGraphFile+".gt")

	minVertEdges = int(getConfig("Graph","minVertEdges"))
	print("minVertEdges = {}".format(minVertEdges))

	if minVertEdges > 0:
		# load kdtree with distances --> edges
		print("loading kdtree.")
		kdtreeFile = getConfig("Files","kdtree")
		with open(kdtreeFile, 'rb') as f:
			kdtree = pickle.load(f)

		# query kdtree for minVertEdges NN
		nearest_ind_k = kdtree.query(nonZeroData, k=minVertEdges+1, return_distance=False)
		
		edge_list = []

		# for every isolated vertex
		for u in graph.vertices:
			if (u.out_degree() == 0):
				# not so sure about [u] below
				edge_list.append((u, nearest_ind_k[u][1]))
				# edge_dists.append()
		
		edge_list = np.array(edge_list)
		print("edge loop done.\n")

		graph.add_edge_list(edge_list)
		print("{} edges added to graph.\n".format(len(edge_list)))

		# save graph
		outputPath = getConfig("Paths","outputPath")
		connGraphFile = getConfig("Files","connectedGraph")
		print("saving graph (file prefix: {}{}).\n".format(outputPath, connGraphFile))
		graph.save(outputPath+connGraphFile+".gt", fmt="gt")
		graph.save(outputPath+connGraphFile+".xml", fmt="xml")

def parallelizeCube():
	# TODO
	# calls getSplitData()
	return

def exportEdges():
	# load graph
	print("loading graph.")
	baseGraphFile = getConfig("Graph","base_graph")
	base_graph = load_graph(baseGraphFile+".gt")

	# create dataframe with edges
	print("creating df with edges.")
	edgesDF = pd.DataFrame(base_graph.edges(), columns=["node1","node2"])

	# save dataframe
	outputPath = getConfig("Paths","outputPath")
	edgesFile = getConfig("Files","edgesFile")
	edgesDF.to_csv(outputPath+edgesFile, index=False)

def getUniqueDataFrame():
	# load graph
	print("loading graph.")
	baseGraphFile = getConfig("Graph","base_graph")
	base_graph = load_graph(baseGraphFile+".gt")

	# create dataframe with edges
	print("creating unique dataframe.")
	uniqueDF = pd.DataFrame(base_graph.edges(), columns=["node1","node2"])

	# get vertices' positions
	print("loading dataframe with positions.")
	df_vertices = getPandasDF()

	print("assigning positions to vertices.")
	posDF = df_vertices[['x', 'y', 'z']]
	# posTuples = [tuple(t) for t in posDF.values]


	# posDic = {i: t for i, t in enumerate(posDF.values)}
	posDic = {i: ",".join(map(str, t)) for i, t in enumerate(posDF.values)}

	uniqueDF.replace({"node1": posDic, "node2": posDic})

	print(uniqueDF.head())
	print(uniqueDF.describe())

	# save dataframe
	outputPath = getConfig("Paths","outputPath")
	uniqueFile = getConfig("Files","uniqueDF")
	uniqueDF.to_csv(outputPath+uniqueFile, index=False)

def getGraphPartition():
	# load graph
	print("loading graph.")
	baseGraphFile = getConfig("Graph","base_graph")
	base_graph = load_graph(baseGraphFile+".gt")

	# get graph partition
	print("minimizing to get partition.")
	state = minimize_blockmodel_dl(base_graph)
	outputPath = getConfig("Paths","outputPath")
	partitionGraph = getConfig("Files","partitionGraph")

	# save graph plot
	print("saving plot.")
	# NEED TO INCORPORATE POSITIONS.
	state.draw(output=outputPath+partitionGraph)

def getHyperGraph():
	baseGraphFile = getConfig("Graph","base_graph")
	base_graph = load_graph(baseGraphFile+".gt")

##### MAIN FUNCTION #####
def main():
	if isOperationSet(operation="createDataFiles"):
		createDataFiles()
	if isOperationSet(operation="createGraph"):
		graph = createGraph()
		if isOperationSet(operation="connectVertices"):
			graph = connectVertices(graph)
	else:
		if isOperationSet(operation="connectVertices"):
			graph = connectVertices()
	if isOperationSet(operation="exportEdges"):
		exportEdges()
	if isOperationSet(operation="parallelizeCube"):
		parallelizeCube()
	if isOperationSet(operation="getGraphPartition"):
		getGraphPartition()
	if isOperationSet(operation="getUniqueDataFrame"):
		getUniqueDataFrame()

if __name__ == "__main__":
	main()
