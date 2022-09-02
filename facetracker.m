clear all;
close all;
clc;

%videoReader = vision.VideoFileReader('tilted_face.avi');
mycam = webcam(2);

%% Create Video Player
%videoPlayer = vision.VideoPlayer;
%% Read first frame
%videoFrame = step(videoReader);
videoFrame = snapshot(mycam);
figure;imshow(videoFrame);title('Input Frame');

%% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
bbox = step(faceDetector,videoFrame);
figure;imshow(videoFrame);title('Detected Object');hold on;
rectangle('Position',bbox,'LineWidth',5,'EdgeColor','r');
% save location of face as a polygon 
x = bbox(1, 1); y = bbox(1, 2); w = bbox(1, 3); h = bbox(1, 4);
bboxPolygon = [x, y, x+w, y, x+w, y+h, x, y+h];


%% Create and Initialize Tracker
% Find points on detected objects
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);

% Display detected feature points
figure;
imshow(videoFrame);
hold on;
plot(points);
title('Detected Features');

%% create point tracker object
tracker = vision.PointTracker;
%initialize tracker with detecte dpoints
initialize(tracker,points.Location,videoFrame);
oldPoints = points.Location;


%% Loop through video
while  (1)
     % get the next frame
    %videoFrame = step(videoReader);
    videoFrame = snapshot(mycam);

    
    % Use pointTracker to track feature points
    [points,validity] = step(tracker,videoFrame);
    
    % Keep only the valid points and discard the rest
    visiblePoints = points(validity,:);
    oldInliers = oldPoints(validity,:);
    
    % Save the new state of the point tracker
    oldPoints = visiblePoints;
    if size(oldPoints, 1) == 0
        h = msgbox('A face foi perdida');
        break;
    else
        setPoints(tracker, oldPoints);

        if size(visiblePoints, 1) >= 2 

            % filter outliers using RANSAC
            [xform, oldInliers, visiblePoints]...
                = estimateGeometricTransform(oldInliers, visiblePoints,...
                                            'similarity', 'MaxDistance', 4);


            % Find bounding box by applying the transformation to the bounding box 
            % from previous frame to current frame to
            [bboxPolygon(1:2:end), bboxPolygon(2:2:end)] ...
                = transformPointsForward(xform, bboxPolygon(1:2:end), bboxPolygon(2:2:end));

            % Insert the bounding box around the object being tracked
            videoFrame = insertShape(videoFrame, 'FilledPolygon', bboxPolygon,'Opacity',0.4);


            % Reset the points
            oldPoints = visiblePoints;
            setPoints(tracker, oldPoints);

        end

        %Display tracked points 
        out = insertMarker(videoFrame, visiblePoints, '+','Color', 'white');
        mean(visiblePoints( 1))
        mean(visiblePoints( 2))
        %end of video processing code

        imshow(out);
        % Display output
        %step(videoPlayer, out);
    end
    
end

%% release video reader and writer
close all;
clear all;
clc;
