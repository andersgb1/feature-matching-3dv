#!/usr/bin/env python

import argparse, itertools, numpy, os, timeit

import sys
sys.path.append('../covis/build/lib')

from covis import *

# Setup program options
po = argparse.ArgumentParser()

# po.add_argument("--root", default="/media/andersgb1/sdur-2/data/abuch/datasets/3dv", type=str, help="root path of your dataset")
# po.add_argument("--root", default="/home/andersgb1/workspace/datasets/3dv", type=str, help="root path of your dataset")
po.add_argument("--root", default="./3dv", type=str, help="root path of your dataset")

# po.add_argument("--scene-dir", '-s', default="scenes/decimated_0.25", type=str, help="subdirectory for the scene models")
po.add_argument("--scene-dir", '-s', default="scenes/decimated_0.25_reduced", type=str, help="subdirectory for the scene models")

po.add_argument("--object-dir", '-o', default="models/dec_0.05", type=str, help="subdirectory for the object models")
po.add_argument("--pose-dir", '-p', default="ground_truth", type=str, help="subdirectory for the ground truth pose models")
po.add_argument("--object-ext", default="", type=str, help="object file extension")
po.add_argument("--scene-ext", default="", type=str, help="scene file extension")
po.add_argument("--pose-ext", default="", type=str, help="pose file extension")
po.add_argument("--pose-sep", default="", type=str, help="pose file separator")
po.add_argument("--object-regex", default="", type=str, help="set this option to use a regular expression search when collecting object files")
po.add_argument("--scene-regex", default="", type=str, help="set this option to use a regular expression search when collecting scene files")
po.add_argument("--scene-nth", default=1, type=int, help="set this value to > 1 to process every n'th scene")
po.add_argument("--scene-offset", default=0, type=int, help="set the start scene index")

po.add_argument("--resolution-surface", default=1, type=float, help="downsample point clouds to this resolution (<= 0 for the average object resolution)")
po.add_argument("--radius-normal", default=5, type=float, help="normal estimation radius as a multiplum of the resolution")
po.add_argument("--resolution-feature", default=10, type=float, help="resolution of features in mr")
po.add_argument("--feature", '-f', default="ecsad", type=str, help="name the feature to compute - possible names are: ecsad,fpfh,ndhist,ppfhist,ppfhistfull,si,usc")
po.add_argument("--radius-feature", '-r', default=0.25, type=float, help="feature estimation radius - if > 1 as a multiplum of the resolution, else if < 1 as a multiplum of the average object diagonal")

po.add_argument("--knn", '-k', default=0, type=int, help="specifiy number of neighbors to search for during matching - the default (0) instead uses Lowe's ratio matching")
# po.add_argument("--match-scene-object", action='store_true', help="set to true to invert the matching direction and use match from object features to scene features")

po.add_argument("--inlier-threshold", default=0, type=float, help="specify inlier threshold in mr (set to <= 0 to use feature resolution)")

po.add_argument("--output-dir", default="output", type=str, help="specify output directory (leave empty for no outputs)")
po.add_argument('--resume', action='store_true', help="set to true to avoid overwriting output files")

args = po.parse_args()

print('Loading dataset from root {}...'.format(args.root))
dataset = util.loadDataset(args.root, args.object_dir, args.scene_dir, args.pose_dir, args.object_ext, args.scene_ext, args.pose_ext, args.pose_sep, args.object_regex, args.scene_regex)
print('\tGot {} model(s), {} scenes and {} poses'.format(len(dataset.objectLabels), len(dataset.sceneLabels), len(dataset.poseLabels)))
objects = dataset.objects

# Surface scaling
resolution = args.resolution_surface
resolutionInput = resolution > 0
if not resolutionInput:
    resolution = 0
diag = 0
for i in range(len(objects)):
    if not resolutionInput:
        resolution += detect.computeResolution(objects[i])
    diag += detect.computeDiagonal(objects[i])
if not resolutionInput:
    resolution /= len(objects)
diag /= len(objects)
frad = args.radius_feature * resolution if args.radius_feature > 1 else args.radius_feature * diag
fres = args.resolution_feature * resolution

objectSurf = []
objectCloud = []
sys.stdout.write('Preprocessing '),sys.stdout.flush()

startTime = timeit.default_timer()

for i in range(len(objects)):
    sys.stdout.write(dataset.objectLabels[i] + ' '),sys.stdout.flush()
    objectSurf.append(filter.preprocess(mesh=objects[i],
                                        resolution=resolution,
                                        normalRadius=args.radius_normal * resolution,
                                        orientNormals=True))

    # Generate feature points
    objectCloud.append(filter.downsample(cloud=objectSurf[i], resolution=fres))
sys.stdout.write('\n'),sys.stdout.flush()

print('\tPreprocessing time: {:.3f} seconds'.format(timeit.default_timer() - startTime))


# Compute features
print('Computing {} object features with a radius of {:.3f}...'.format(sum([c.width * c.height for c in objectCloud]), frad))
objectFeat = feature.computeFeature(name=args.feature,
                                    clouds=objectCloud,
                                    surfaces=objectSurf,
                                    radius=frad)

# Loop over scenes
totalPositives = 0
totalRetrieved = 0
totalInliers = 0
for i in range(args.scene_offset, dataset.size, args.scene_nth):
    print('Processing scene {}/{} ({})...'.format(i+1, dataset.size, scene.label))
    if dataset.empty(i):
        print('\tScene empty - skipping...')
        continue

    scene = dataset.at(i)
    
    outputFile = args.output_dir + '/' + args.feature + '{}'.format(args.radius_feature) + '---' + scene.label + '---' + dataset.objectLabels[0] + '.txt'
    if args.resume and os.path.isfile(outputFile):
        print('\tScene {} already processed - skipping...'.format(scene.label))
        continue;

    print('Preprocessing scene...')
    sceneMesh = scene.scene
    sceneSurf = filter.preprocess(mesh=sceneMesh,
                                  resolution=resolution,
                                  normalRadius=args.radius_normal * resolution)

    sceneCloud = filter.downsample(cloud=sceneSurf, resolution=fres)

    print('Computing {} scene features...'.format(sceneCloud.size))
    sceneFeat = feature.computeFeature(name=args.feature,
                                       cloud=sceneCloud,
                                       surface=sceneSurf,
                                       radius=frad)

    # if args.match_scene_object:
    #     query = sceneFeat
    #     target = objectFeat
    # else:
    query = objectFeat
    target = sceneFeat

    startTime = timeit.default_timer()
    if args.knn <= 0:
        print('Using ratio matcher for {} --> {} {} features (dim={})...'.format(query.shape[1], target.shape[1], args.feature, query.shape[0]))
        featureCorr = detect.computeRatioMatchesExact(query, target)
    else:
        assert args.knn <= target.shape[1]
        print('Using k-NN (k={}) matcher for {} --> {} {} features (dim={})...'.format(args.knn, query.shape[1], target.shape[1], args.feature, query.shape[0]))
        featureCorr = detect.computeKnnMatchesExact(query, target, args.knn)
    featureCorr = core.flatten(featureCorr)
    featureCorr = core.sort(featureCorr)
    print('\tMatching time: {:.3f} seconds'.format(timeit.default_timer() - startTime))

    print('Computing visibility of object(s) in scene...')
    startTime = timeit.default_timer()
    objectMask = scene.objectMask
    objectCloudAll = core.PointCloud()
    idxVisible = 0
    for j in range(len(objectCloud)):
        if objectMask[j]:
            tmp = core.transform(objectCloud[j], scene.poses[idxVisible].array())
            idxVisible += 1
        else:
            tmp = core.PointCloud(size=objectCloud[j].size)
        objectCloudAll += tmp

    # if args.match_scene_object:
    #     queryCloud = sceneCloud
    #     targetCloud = objectCloudAll
    # else:
    queryCloud = objectCloudAll
    targetCloud = sceneCloud

    # Utility function for computing the quality (Euclidean fit) of each correspondence
    def alignments(correspondences, query, target):
        iq, it = [c.query for c in correspondences], [c.match[0] for c in correspondences]
        cq, ct = query[:, iq], target[:, it]
        return numpy.linalg.norm(ct - cq, axis=0)

    # Resolve inlier threshold
    inlierThreshold = resolution * (args.resolution_feature if args.inlier_threshold <= 0 else args.inlier_threshold)

    # Compute visibility
    print('\tComputing surface correspondences...')
    corrTotal = core.computeAllCorrespondences(query.shape[1], target.shape[1], distance=-1)
    corrTotalDist = alignments(corrTotal, queryCloud.array(), targetCloud.array())
    positives = list(itertools.compress(corrTotal, corrTotalDist <= inlierThreshold))

    # Gather number of positives/retrieved
    numPositives = len(core.unique(positives, max(1, args.knn))) # Use the max to handle the ratio case

    numRetrieved = len(featureCorr)
    totalPositives += numPositives
    totalRetrieved += numRetrieved
    print('\tVisibility computation time: {:.3f} seconds'.format(timeit.default_timer() - startTime))

    print('Verifying {} matches with an inlier threshold of {}...'.format(len(featureCorr), inlierThreshold))
    startTime = timeit.default_timer()
    featureCorrDist = alignments(featureCorr, queryCloud.array(), targetCloud.array())
    inliers = list(itertools.compress(featureCorr, featureCorrDist <= inlierThreshold))

    numInliers = len(inliers)
    totalInliers += numInliers
    print('\tVerification time: {:.3f} seconds'.format(timeit.default_timer() - startTime))

    if len(args.output_dir):
        assert len(dataset.objectLabels) == 1
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        outputFile = args.output_dir + '/' + args.feature + '{}'.format(args.radius_feature) + '---' + scene.label + '---' + dataset.objectLabels[0] + '.txt'
        outputResults = numpy.zeros((len(corrTotal), 3), dtype=float)
        # First mark all possible correspondences with alignment distance and invalid matching distance
        outputResults[:, 0] = corrTotalDist
        outputResults[:, 1] = numpy.nan

        # Then mark which of them we retrieved with a valid matching distance
        for j in range(len(featureCorr)):
            iq = featureCorr[j].query
            tq = featureCorr[j].match[0]
            idx = iq * target.shape[1] + tq
            # outputResults[idx, 1] = 1
            # outputResults[idx, 2] = featureCorr[j].distance[0]
            outputResults[idx, 1] = featureCorr[j].distance[0]
        # We only need to save the retrieved and the positives
        # TODO: Regard all matches closer than some upper bound as positives in the saved results for now
        mask = numpy.bitwise_or(outputResults[:,0] <= max(inlierThreshold, fres, frad), ~numpy.isnan(outputResults[:,1]))
        outputResults = outputResults[mask, :]
        # TODO: Insert a bogus column which counts the number of positives
        outputResults[0, 2] = numPositives

        print('Saving scene results for {} correspondences to output file {}...'.format(outputResults.shape[0], outputFile))
        fid = open(outputFile, 'wb')
        outputResults.tofile(fid)
        fid.close()

    print('Stats for scene {}'.format(scene.label))
    if args.knn <= 0:
        print('\t{} feature matches retrieved'.format(len(featureCorr)))
        print('\t{} positive matches'.format(numPositives))
    else:
        print('\t{} feature matches retrieved (searching for {}-NNs)'.format(len(featureCorr), args.knn))
        print('\t{} positive {}-NN matches'.format(numPositives, args.knn))
    print('\t{} inliers'.format(numInliers))
    
    if numPositives > 0:
        print('\tRecall for scene:    {}/{} ({:.2f} %)'.format(numInliers, numPositives, 100 * numInliers / float(numPositives)))
    if numRetrieved > 0:
        print('\tPrecision for scene: {}/{} ({:.2f} %)'.format(numInliers, numRetrieved, 100 * numInliers / float(numRetrieved)))

    print('Overall stats')
    if totalPositives > 0:
        print('\tRecall:      {}/{} ({:.2f} %)'.format(totalInliers, totalPositives, 100 * totalInliers / float(totalPositives)))
    if totalRetrieved > 0:
        print('\tPrecision:   {}/{} ({:.2f} %)'.format(totalInliers, totalRetrieved, 100 * totalInliers / float(totalRetrieved)))

if totalPositives > 0 and totalRetrieved > 0:
    print('Overall stats')
    print('\tRecall:      {}/{} ({:.2f} %)'.format(totalInliers, totalPositives, 100 * totalInliers / float(totalPositives)))
    print('\tPrecision:   {}/{} ({:.2f} %)'.format(totalInliers, totalRetrieved, 100 * totalInliers / float(totalRetrieved)))
