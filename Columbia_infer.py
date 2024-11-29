import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features

from scipy.io import wavfile
from sklearn.metrics import accuracy_score, f1_score

from ASDnB import ASDnB

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "Columbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="columbia",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="Columbia",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="ASDnB_AVA.model",   help='Path for the pretrained model')

parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
parser.add_argument('--colSavePath',           type=str, default="colDataPath",  help='Path for inputs, tmps and outputs')

parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
parser.add_argument('--batchSize',    type=int,   default=2000,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
parser.add_argument('--nDataLoaderThread', type=int, default=1,  help='Number of loader threads')

args = parser.parse_args()

def create_video(fileName, dir):
	video = cv2.VideoCapture(os.path.join(dir, fileName + '.avi'))
	videoFeature = []
	while video.isOpened():
		ret, frames = video.read()
		if ret == True:
			face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
			face = cv2.resize(face, (224,224))
			face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
			videoFeature.append(face)
		else:
			break
	video.release()
	videoFeature = numpy.array(videoFeature)
	return videoFeature

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained model
	s = ASDnB(**vars(args))
	ckpt = torch.load(args.pretrainModel, map_location='cuda')
	s.load_state_dict(ckpt)

	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	# durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	durationSet = {12,24,48,60} 
	
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)

		videoFeature = create_video(fileName, args.pycropPath)
		videoFeatureBody = create_video(fileName, args.pycropPathBody)
		
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		videoFeatureBody = videoFeatureBody[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use model
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):

					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					inputVB = torch.FloatTensor(videoFeatureBody[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()

					audioEmbed = s.model.forward_audio_frontend(inputA)
					visualEmbed = s.model.forward_visual_frontend(inputV, inputVB)	
					out = s.model.forward_combination_backend(audioEmbed, visualEmbed) 

					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def evaluate_col_ASD(tracks, scores, args):
	txtPath = args.videoFolder + '/col_labels/fusion/*.txt' # Load labels
	predictionSet = {}
	for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
		predictionSet[name] = [[],[]]
	dictGT = {}
	txtFiles = glob.glob("%s"%txtPath)
	for file in txtFiles:
		lines = open(file).read().splitlines()
		idName = file.split('/')[-1][:-4]
		for line in lines:
			data = line.split('\t')
			frame = int(int(data[0]) / 29.97 * 25)
			# Cara
			x1 = int(data[1])
			y1 = int(data[2])
			x2 = int(data[1]) + int(data[3])
			y2 = int(data[2]) + int(data[3])

			gt = int(data[4])
			if frame in dictGT:
				dictGT[frame].append([x1,y1,x2,y2,gt,idName])
			else:
				dictGT[frame] = [[x1,y1,x2,y2,gt,idName]]	
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Load files
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]				
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		if fidx in dictGT: # This frame has label
			for gtThisFrame in dictGT[fidx]: # What this label is ?
				faceGT = gtThisFrame[0:4]
				labelGT = gtThisFrame[4]
				idGT = gtThisFrame[5]
				ious = []
				for face in faces[fidx]: # Find the right face in my result
					# faceLocation_new = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
					faceLocation_new = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
					iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol = True)
					if iou > 0.5:
						ious.append([iou, round(face['score'],2)])
				if len(ious) > 0: # Find my result
					ious.sort()
					labelPredict = ious[-1][1]
				else:					
					labelPredict = 0
				x1 = faceGT[0]
				y1 = faceGT[1]
				width = faceGT[2] - faceGT[0]
				predictionSet[idGT][0].append(labelPredict)
				predictionSet[idGT][1].append(labelGT)
	names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas'] # Evaluate
	names.sort()
	F1s = 0
	for i in names:
		scores = numpy.array(predictionSet[i][0])
		labels = numpy.array(predictionSet[i][1])
		scores = numpy.int64(scores > 0)
		F1 = f1_score(labels, scores)
		ACC = accuracy_score(labels, scores)
		if i != 'abbas':
			F1s += F1
			print("%s, ACC:%.2f, F1:%.2f"%(i, 100 * ACC, 100 * F1))
	print("Average F1:%.2f"%(100 * (F1s / 5)))	  

# Main function
def main():
	
	args.savePath = os.path.join(args.videoFolder, args.videoName)

	# Initialization 
	args.pyaviPath = os.path.join(args.savePath, 'pyavi')
	args.pyframesPath = os.path.join(args.savePath, 'pyframes')
	args.pyworkPath = os.path.join(args.savePath, 'pywork')
	args.pycropPath = os.path.join(args.savePath, 'pycrop_face_far')
	args.pycropPathBody = os.path.join(args.savePath, 'pycrop_body')

	savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
	fil = open(savePath, 'rb')
	vidTracks = pickle.load(fil)

	# Active Speaker Detection
	files = glob.glob("%s/*.avi"%args.pycropPath)
	files.sort()
	scores = evaluate_network(files, args)

	evaluate_col_ASD(vidTracks, scores, args) 

if __name__ == '__main__':
    main()
