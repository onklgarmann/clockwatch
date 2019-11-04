#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector> 
#include <limits>

using namespace std;
using namespace cv;

int lookupBase(int);

int main(int argc, char** argv){
	//Program for å isolere det største objektet i bildet.  Kristoffer Johan Garmann 2019.
    //
    //Tar inn ferdig binært bilde og isolerer største komponent ved connected component labeling.
    //Ikke spesielt raffinert eller effektivt, men funker greit med bilder i vår størrelse innenfor
    //integer limits på RPi.
	if (argc < 3) {
        cerr << "Mangler sti til filer, ./finnViser ./sti-til-inn-filnavn.bmp ./sti-til-ut-filnavn.bmp" << endl;
        return 1;
    }
	Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    Mat map = image.clone();
    map = 0;
    int height = image.rows;
    int width = image.cols;
    vector<vector<int>> mapVector(height, vector<int>(width, 0));
    int nShapes = 0;
    vector<vector<int>> relationships;
    
    for(int i = 0 ; i < image.rows; i++)
    {
        
        for(int j = 0; j < image.cols; j++)
        {
            if ((int)image.at<uchar>(i,j)!=0){
                vector<int> neighbors;
                if(i != 0 && j != 0){
                    neighbors.push_back(mapVector[i-1][j]);
                    neighbors.push_back(mapVector[i][j-1]);
                }
                else if (i == 0 && j != 0){
                    neighbors.push_back(0);
                    neighbors.push_back(mapVector[i][j-1]);
                }
                else if (i != 0 && j == 0){
                    neighbors.push_back(mapVector[i-1][j]);
                    neighbors.push_back(0);
                }
                else{
                    neighbors.push_back(0);
                    neighbors.push_back(0);
                }

                if (neighbors[0]==0 && neighbors[1]==0){
                    nShapes++;
                    mapVector[i][j]=nShapes;
                }
                else if (neighbors[0]==0 || neighbors[1]==0){
                    mapVector[i][j]=*max_element (neighbors.begin(),neighbors.end());
                }
                else if (neighbors[0]!=neighbors[1]){
                    mapVector[i][j]=*min_element (neighbors.begin(),neighbors.end());
                    sort(neighbors.begin(),neighbors.end());
                    relationships.push_back(neighbors);
                    
                }
                else{
                    mapVector[i][j]=neighbors[0];
                }
                
                
            }

        }
        
    }
    //cout << endl << endl << "Labels: " << nShapes << endl << endl;
    vector<vector<int>> refinedRelations(nShapes+1);
    for (int i = 1; i<relationships.size();i++){
        if (find(refinedRelations[relationships[i][0]].begin(), refinedRelations[relationships[i][0]].end(), relationships[i][1]) == refinedRelations[relationships[i][0]].end()){
            refinedRelations[relationships[i][0]].push_back(relationships[i][1]);
        }
        
    }
    vector<int> roots(nShapes+1, 0);

    
    
    for (int i = 1; i<refinedRelations.size();i++){
        //cout << i << "\t: ";
        if (!refinedRelations[i].empty()){
            sort(refinedRelations[i].begin(), refinedRelations[i].end());
            for (int j = 0; j<refinedRelations[i].size();j++){
                int c = refinedRelations[i][j];
                //cout << "rRelations: " << c << " root c:" << roots[c] << " root i: " << roots[i] << " etter: ";
                if (roots[c]==0 && roots[i]==0){
                    roots[c]=i;
                }
                else if (roots[c]!=0 && roots[i]==0){
                    roots[i]=roots[c];
                }
                else if (roots[c]==0 && roots[i]!=0){
                    roots[c]=roots[i];
                }
                else if (roots[c]<roots[i]){
                    
                    cout << i << ":\t rootc: " << roots[c] << " rooti: " << roots[i] << endl;
                    roots[i]=roots[c];
                    i = roots[i];
                }
            
                //cout << "rRelations: " << c << " root c:" << roots[c] << " root i: " << roots[i] << endl;
            }
        } 
        
    }

/*
    for (int i = 1; i<refinedRelations.size();i++){
        cout << i << ":  \t" << roots[i] << ":  ";
        if (!refinedRelations[i].empty()){
            
            for (int j = 0; j<refinedRelations[i].size();j++){
                cout << refinedRelations[i][j] << " ";
            }
            
        }
        cout << endl;
    }
*/



    vector<long int> rootSize(nShapes+1, 0);

    for(int i = 0 ; i < height; i++)
    {
        
        for(int j = 0; j < width; j++)
        {
            int pixel = mapVector[i][j];
            if (pixel != 0 && roots[pixel] == 0){
                rootSize[pixel]++;
            }
            else if (pixel != 0){
                //cout << i << ", " << j << "\t: " << pixel << "\t: " << roots[pixel] << " rootsize: " << rootSize[roots[pixel]] << endl;
                mapVector[i][j]=roots[pixel];
                rootSize[roots[pixel]]++;
            
                
            }
        }
       
    }
    
    
    

    int viser = max_element(rootSize.begin(),rootSize.end()) - rootSize.begin(); 
    //rootSize[viser]=0;
     //viser = max_element(rootSize.begin(),rootSize.end()) - rootSize.begin(); 
    //cout << "Largest element: " << viser << '\t' << *max_element(rootSize.begin(),rootSize.end()) << endl;
    
    for(int i = 0 ; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            map.at<uchar>(i,j)= mapVector[i][j];
        }
    }
    
    for(int i = 0 ; i < height; i++)
    {
        
        for(int j = 0; j < width; j++)
        {
            int pixel = mapVector[i][j];
            //cout << i << " : " << j << " : " << pixel << " : ";
            if (pixel !=0 && pixel != viser ){
                map.at<uchar>(i,j) = 0;
                //cout << 0;
            }
            else if (pixel == viser){
                map.at<uchar>(i,j) = 255;
                //cout << 255;
            }
            //cout << endl;
        }
    }

    cout << nShapes << '\t' << viser << '\t' << rootSize[124];
    
    imwrite(argv[2], map);
    
    cout << endl << "finnViser completed" << endl;
}