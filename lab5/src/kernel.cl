__kernel void convolution(int filterWidth, __constant float *filter, int imageHight, int imageWidth, __global float *inputImage, __global float *outputImage) 
{
   int halfilterSize = filterWidth / 2;
   float sum;
   int x = get_global_id(0);
   int y = get_global_id(1);
   int k, l;

   sum = 0;
   for(k = -halfilterSize; k <= halfilterSize; k++){
           for(l = -halfilterSize; l <= halfilterSize; l++){
               if(y + k >= 0 && y + k < imageHight && x + l >= 0 && x + l < imageWidth){
               sum += inputImage[(y + k) * imageWidth + x + l] * filter[(k + halfilterSize) * filterWidth + l + halfilterSize];
               }
            }
    }
    outputImage[y * imageWidth + x] = sum;
}
