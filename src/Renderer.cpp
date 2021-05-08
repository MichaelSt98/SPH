#include "../include/Renderer.h"

Renderer::Renderer(const int _numParticles, const int _width, const int _height, const double _renderScale, const double _maxVelocityColor,
                   const double _minVelocityColor, const double _particleBrightness, const double _particleSharpness,
                   const int _dotSize, const double _systemSize, const int _renderInterval) :
                        numParticles { _numParticles },
                        width { _width }, height { _height }, renderScale { _renderScale },
                        maxVelocityColor { _maxVelocityColor }, minVelocityColor { _minVelocityColor },
                        particleBrightness { _particleBrightness },
                        particleSharpness { _particleSharpness }, dotSize { _dotSize },
                        systemSize { _systemSize }, renderInterval { _renderInterval } {

    LOGCFG.headers = true;
    LOGCFG.level = INFO;

    Logger(INFO) << "RENDERING RELATED PARAMETERS";
    Logger(INFO) << "--------------------------------------";
    Logger(INFO) << "num Particles:      " << numParticles;
    Logger(INFO) << "width:              " << width;
    Logger(INFO) << "height:             " << height;
    Logger(INFO) << "renderScale:        " << renderScale;
    Logger(INFO) << "maxVelocityColor:   " << maxVelocityColor;
    Logger(INFO) << "minVelocityColor:   " << minVelocityColor;
    Logger(INFO) << "particleBrightness: " << particleBrightness;
    Logger(INFO) << "particleSharpness:  " << particleSharpness;
    Logger(INFO) << "dotSize:            " << dotSize;
    Logger(INFO) << "systemSize:         " << systemSize;
    Logger(INFO) << "renderInterval:     " << renderInterval;
    Logger(INFO) << "--------------------------------------";

}

int Renderer::getRenderInterval() {
    return renderInterval;
}

void Renderer::createFrame(char* image, double* hdImage, Body* b, int step)
{
    Logger(INFO) <<  "Writing frame " << step;


    Logger(DEBUG) << "Clearing Pixels ...";
    renderClear(image, hdImage);


    Logger(DEBUG) << "Rendering Particles ...";
    renderBodies(b, hdImage);


    Logger(DEBUG) << "Writing frame to file ...";
    writeRender(image, hdImage, step);
}

void Renderer::renderClear(char* image, double* hdImage)
{
    memset(image, 0, width*height*3);
    memset(hdImage, 0, width*height*3*sizeof(double));
}

void Renderer::renderBodies(Body* b, double* hdImage)
{
    for(int index=0; index<numParticles; index++)
    {
        Body *current = &b[index];

        int x = toPixelSpace(current->position.x, width);
        int y = toPixelSpace(current->position.y, height);

        if (x>dotSize && x<width-dotSize &&
            y>dotSize && y<height-dotSize)
        {
            double vMag = current->velocity.magnitude(); //magnitude(current->velocity);
            colorDot(current->position.x, current->position.y, vMag, hdImage);

            /**
            for (int i_x = int(current->position.x*100 - ((renderScale*systemSize)*100)/200); i_x < int(current->position.x*100 + ((renderScale*systemSize)*100)/200); i_x++) {
                for (int i_y = int(current->position.x*100 - ((renderScale*systemSize)*100)/200); i_y < int(current->position.x*100 + ((renderScale*systemSize)*100)/200); i_y++) {
                    colorDot(i_x/100.0, i_y/100.0, vMag, hdImage);
                }
            }
            **/
        }
    }
}

double Renderer::toPixelSpace(double p, int size)
{
    return (size/2.0) * (1.0 + p/(systemSize*renderScale));
}

void Renderer::colorDot(double x, double y, double vMag, double* hdImage)
{
    const double velocityMax = maxVelocityColor;
    const double velocityMin = minVelocityColor; //0.0;

    if (vMag < velocityMin)
        return;

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
            colorAt(int(xP+i),int(yP+j), c, cFactor, hdImage);
        }
    }

}

void Renderer::colorAt(int x, int y, const color& c, double f, double* hdImage)
{
    int pix = 3*(x+width*y);
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

    for (int i=0; i<width*height*3; i++)
    {
        data[i] = int(255.0*clamp(hdImage[i]));
    }

    int frame = step/renderInterval + 1;//renderInterval;
    char name[128];
    sprintf(name, "images/Step%05i.ppm", frame);
    std::ofstream file (name, std::ofstream::binary);

    if (file.is_open())
    {
        file << "P6\n" << width << " " << height << "\n" << "255\n";
        file.write(data, width*height*3);
        file.close();
    }

}


