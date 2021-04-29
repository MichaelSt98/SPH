#include "../include/Renderer.h"


Renderer::Renderer(const int _numParticles, const int _width, const int _height, const int _depth, const double _renderScale, const double _maxVelocityColor,
                   const double _minVelocityColor, const double _particleBrightness, const double _particleSharpness,
                   const int _dotSize, const double _systemSize, const int _renderInterval) :
                        numParticles { _numParticles },
                        width { _width }, height { _height }, depth { _depth }, renderScale { _renderScale },
                        maxVelocityColor { _maxVelocityColor }, minVelocityColor { _minVelocityColor },
                        particleBrightness { _particleBrightness },
                        particleSharpness { _particleSharpness }, dotSize { _dotSize },
                        systemSize { _systemSize }, renderInterval { _renderInterval } {

}

Renderer::Renderer(ConfigParser &confP) : Renderer(
        confP.getVal<int>("numParticles"),
        confP.getVal<int>("width"),
        confP.getVal<int>("height"),
        confP.getVal<int>("depth"),
        confP.getVal<double>("renderScale"),
        confP.getVal<double>("maxVelColor"),
        confP.getVal<double>("minVelColor"),
        confP.getVal<double>("particleBrightness"),
        confP.getVal<double>("particleSharpness"),
        confP.getVal<int>("dotSize"),
        confP.getVal<double>("systemSize"),
        confP.getVal<int>("renderInterval")) {

}

int Renderer::getRenderInterval() {
    return renderInterval;
}

void Renderer::createFrame(char* image, double* hdImage, Particle* p, int step, Domain* box)
{
    Logger(DEBUG) <<  "Writing frame " << step;


    Logger(DEBUG) << "Clearing Pixels ...";
    renderClear(image, hdImage);

    Logger(DEBUG) << "Drawing Domain-Box ...";
    drawDomainBox(box, hdImage);

    Logger(DEBUG) << "Rendering Particles ...";
    renderBodies(p, hdImage);


    Logger(DEBUG) << "Writing frame to file ...";
    writeRender(image, hdImage, step);
}

void Renderer::createFrame(char* image, double* hdImage, Particle* p, int* processNum, int numprocs, int step, Domain* box)
{
    Logger(DEBUG) <<  "Writing frame " << step;


    Logger(DEBUG) << "Clearing Pixels ...";
    renderClear(image, hdImage);

    Logger(DEBUG) << "Drawing Domain-Box ...";
    drawDomainBox(box, hdImage);

    Logger(DEBUG) << "Rendering Particles ...";
    renderBodies(p, processNum, numprocs, hdImage);

    Logger(DEBUG) << "Writing frame to file ...";
    writeRender(image, hdImage, step);
}

void Renderer::renderClear(char* image, double* hdImage)
{
    memset(image, 0, width*2*height*3);
    memset(hdImage, 0, width*2*height*3*sizeof(double));
}

void Renderer::drawDomainBox(Domain* box, double* hdImage){

    const int linew = 4;

    /** draw x-y-plane **/
    int xlower, xupper, ylower, yupper;

    xlower = toPixelSpace(box->lower[0], width);
    xupper = toPixelSpace(box->upper[0], width);
    ylower = toPixelSpace(box->lower[1], height);
    yupper = toPixelSpace(box->upper[1], height);

    // draw along the x-axis
    for (int x=xlower; x<=xupper; x++){
        // draw lower line
        for (int y=ylower-linew/2;y<=ylower+linew/2;y++){
            int pix = 3*(x+2*width*y);
            hdImage[pix+0] = 1.;
            hdImage[pix+1] = 0.;
            hdImage[pix+2] = 0.;
        }
        // draw upper line
        for (int y=yupper-linew/2;y<=yupper+linew/2;y++){
            int pix = 3*(x+2*width*y);
            hdImage[pix+0] = 1.;
            hdImage[pix+1] = 0;
            hdImage[pix+2] = 0;
        }
    }

    // draw along the y-axis
    for (int y=ylower; y<=yupper; y++){
        // draw left line
        for (int x=xlower-linew/2;x<=xlower+linew/2;x++){
            int pix = 3*(x+2*width*y);
            hdImage[pix+0] = 1.;
            hdImage[pix+1] = 0.;
            hdImage[pix+2] = 0.;
        }
        // draw right line
        for (int x=xupper-linew/2;x<=xupper+linew/2;x++){
            int pix = 3*(x+2*width*y);
            hdImage[pix+0] = 1.;
            hdImage[pix+1] = 0;
            hdImage[pix+2] = 0;
        }
    }

    /** draw x-z-plane **/
    int zlower, zupper;

    zlower = toPixelSpace(box->lower[2], depth);
    zupper = toPixelSpace(box->upper[2], depth);

    // draw along the x-axis
    for (int x=xlower; x<=xupper; x++){
        // draw lower line
        for (int z=zlower-linew/2;z<=zlower+linew/2;z++){
            int pix = 3*(x+2*width*z+width);
            hdImage[pix+0] = 1.;
            hdImage[pix+1] = 0.;
            hdImage[pix+2] = 0.;
        }
        // draw upper line
        for (int z=zupper-linew/2;z<=zupper+linew/2;z++){
            int pix = 3*(x+2*width*z+width);
            hdImage[pix+0] = 1.;
            hdImage[pix+1] = 0;
            hdImage[pix+2] = 0;
        }
    }

    // draw along the y-axis
    for (int z=zlower; z<=zupper; z++){
        // draw left line
        for (int x=xlower-linew/2;x<=xlower+linew/2;x++){
            int pix = 3*(x+2*width*z+width);
            hdImage[pix+0] = 1.;
            hdImage[pix+1] = 0.;
            hdImage[pix+2] = 0.;
        }
        // draw right line
        for (int x=xupper-linew/2;x<=xupper+linew/2;x++){
            int pix = 3*(x+2*width*z+width);
            hdImage[pix+0] = 1.;
            hdImage[pix+1] = 0;
            hdImage[pix+2] = 0;
        }
    }

}

void Renderer::renderBodies(Particle* p, double* hdImage)
{
    /** draw x-y-plane **/
    for(int index=0; index<numParticles; index++)
    {
        Particle *current = &p[index];

        int x = toPixelSpace(current->x[0], width);
        int y = toPixelSpace(current->x[1], height);

        if (x>dotSize && x<width-dotSize &&
            y>dotSize && y<height-dotSize)
        {
            // vMag needed for coloring
            double vMag = 0.;
            for (int d=0; d<DIM; d++){
                vMag += current->v[d] * current->v[d];
            }
            vMag = sqrt(vMag);

            int i2fPrec = 100; // TODO: rename

            for (int i_x = int(current->x[0]*i2fPrec - ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec));
                i_x < int(current->x[0]*i2fPrec + ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec)); i_x++) {
                for (int i_y = int(current->x[1]*i2fPrec - ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec));
                    i_y < int(current->x[1]*i2fPrec + ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec)); i_y++) {
                    colorDotXY(i_x/(double)i2fPrec, i_y/(double)i2fPrec, vMag, hdImage);
                }
            }
        }
    }

    /** draw x-z-plane **/
    for(int index=0; index<numParticles; index++)
    {
        Particle *current = &p[index];

        int x = toPixelSpace(current->x[0], width);
        int z = toPixelSpace(current->x[2], depth);

        if (x>dotSize && x<width-dotSize &&
            z>dotSize && z<depth-dotSize)
        {
            // vMag needed for coloring
            double vMag = 0.;
            for (int d=0; d<DIM; d++){
                vMag += current->v[d] * current->v[d];
            }
            vMag = sqrt(vMag);

            int i2fPrec = 100; // TODO: rename

            for (int i_x = int(current->x[0]*i2fPrec - ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec));
                 i_x < int(current->x[0]*i2fPrec + ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec)); i_x++) {
                for (int i_z = int(current->x[2]*i2fPrec - ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec));
                     i_z < int(current->x[2]*i2fPrec + ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec)); i_z++) {
                    colorDotXZ(i_x/(double)i2fPrec, i_z/(double)i2fPrec, vMag, hdImage);
                }
            }
        }
    }

}

void Renderer::renderBodies(Particle* p, int* processNum, int numprocs, double* hdImage)
{
    /** draw x-y-plane **/
    for(int index=0; index<numParticles; index++)
    {
        Particle *current = &p[index];

        int x = toPixelSpace(current->x[0], width);
        int y = toPixelSpace(current->x[1], height);

        if (x>dotSize && x<width-dotSize &&
            y>dotSize && y<height-dotSize)
        {

            double vMag = minVelocityColor + (maxVelocityColor-minVelocityColor)/(numprocs - 1) * processNum[index];

            int i2fPrec = 100; // TODO: rename

            for (int i_x = int(current->x[0]*i2fPrec - ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec));
                 i_x < int(current->x[0]*i2fPrec + ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec)); i_x++) {
                for (int i_y = int(current->x[1]*i2fPrec - ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec));
                     i_y < int(current->x[1]*i2fPrec + ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec)); i_y++) {
                    colorDotXY(i_x/(double)i2fPrec, i_y/(double)i2fPrec, vMag, hdImage);
                }
            }
        }
    }

    /** draw x-z-plane **/
    for(int index=0; index<numParticles; index++)
    {
        Particle *current = &p[index];

        int x = toPixelSpace(current->x[0], width);
        int z = toPixelSpace(current->x[2], depth);

        if (x>dotSize && x<width-dotSize &&
            z>dotSize && z<depth-dotSize)
        {
            double vMag = minVelocityColor + (maxVelocityColor-minVelocityColor)/(numprocs - 1) * processNum[index];

            int i2fPrec = 100; // TODO: rename

            for (int i_x = int(current->x[0]*i2fPrec - ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec));
                 i_x < int(current->x[0]*i2fPrec + ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec)); i_x++) {
                for (int i_z = int(current->x[2]*i2fPrec - ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec));
                     i_z < int(current->x[2]*i2fPrec + ((renderScale*systemSize)*i2fPrec)/(2*i2fPrec)); i_z++) {
                    colorDotXZ(i_x/(double)i2fPrec, i_z/(double)i2fPrec, vMag, hdImage);
                }
            }
        }
    }

}


double Renderer::toPixelSpace(double p, int size)
{
    return (size/2.0) * (1.0 + p/(systemSize*renderScale));
}

void Renderer::colorDotXY(double x, double y, double vMag, double* hdImage)
{
    const double velocityMax = maxVelocityColor;
    const double velocityMin = minVelocityColor; //0.0;

    if (vMag < velocityMin){
        Logger(ERROR) << "In colorDot(): vMag < 0! This should not happen.";
        return;
    }

    const double vPortion = sqrt((vMag-velocityMin) / velocityMax);

    color c;
    c.r = clamp(4*(vPortion-0.333));
    c.g = clamp(fmin(4*vPortion,4.0*(1.0-vPortion)));
    c.b = clamp(4*(0.5-vPortion));

    for (int i=-dotSize/2; i<dotSize/2; i++)
    {
        for (int j=-dotSize/2; j<dotSize/2; j++)
        {
            double xP = floor(toPixelSpace(x, width));
            double yP = floor(toPixelSpace(y, height));
            double cFactor = particleBrightness /
                             (pow(exp(pow(particleSharpness*
                                          (xP+i-toPixelSpace(x, width)),2.0))
                                  + exp(pow(particleSharpness*
                                            (yP+j-toPixelSpace(y, height)),2.0)),/*1.25*/0.75)+1.0);
            colorAtXY(int(xP+i),int(yP+j), c, cFactor, hdImage);
        }
    }

}

void Renderer::colorDotXZ(double x, double z, double vMag, double* hdImage)
{
    const double velocityMax = maxVelocityColor;
    const double velocityMin = minVelocityColor; //0.0;

    if (vMag < velocityMin){
        Logger(ERROR) << "In colorDot(): vMag < 0! This should not happen.";
        return;
    }

    const double vPortion = sqrt((vMag-velocityMin) / velocityMax);

    color c;
    c.r = clamp(4*(vPortion-0.333));
    c.g = clamp(fmin(4*vPortion,4.0*(1.0-vPortion)));
    c.b = clamp(4*(0.5-vPortion));

    for (int i=-dotSize/2; i<dotSize/2; i++)
    {
        for (int j=-dotSize/2; j<dotSize/2; j++)
        {
            double xP = floor(toPixelSpace(x, width));
            double zP = floor(toPixelSpace(z, depth));
            double cFactor = particleBrightness /
                             (pow(exp(pow(particleSharpness*
                                          (xP+i-toPixelSpace(x, width)),2.0))
                                  + exp(pow(particleSharpness*
                                            (zP+j-toPixelSpace(z, depth)),2.0)),/*1.25*/0.75)+1.0);
            colorAtXZ(int(xP+i),int(zP+j), c, cFactor, hdImage);
        }
    }

}

void Renderer::colorAtXY(int x, int y, const color& c, double f, double* hdImage)
{
    int pix = 3*(x+2*width*y);
    hdImage[pix+0] += c.r*f;
    hdImage[pix+1] += c.g*f;
    hdImage[pix+2] += c.b*f;
}

void Renderer::colorAtXZ(int x, int z, const color& c, double f, double* hdImage)
{
    int pix = 3*(x+2*width*z+width);
    hdImage[pix+0] += c.r*f;
    hdImage[pix+1] += c.g*f;
    hdImage[pix+2] += c.b*f;
}

unsigned char Renderer::colorDepth(unsigned char x, unsigned char p, double f)
{
    return fmax(fmin((x*f+p),255),0);
}

double Renderer::clamp(double x)
{
    return fmax(fmin(x,1.0),0.0);
}

void Renderer::writeRender(char* data, double* hdImage, int step)
{

    for (int i=0; i<width*2*height*3; i++)
    {
        data[i] = int(255.0*clamp(hdImage[i]));
    }

    int frame = step/renderInterval + 1;
    char name[128];
    sprintf(name, "images/Step%05i.ppm", frame);

    std::ofstream file (name, std::ofstream::binary);

    if (file.is_open())
    {
        file << "P6\n" << 2*width << " " << height << "\n" << "255\n";
        file.write(data, width*2*height*3);
        file.close();
    }

}


