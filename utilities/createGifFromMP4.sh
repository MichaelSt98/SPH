ffmpeg -i movie.mp4  -vf "fps=5,scale=500:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 10 - -loop 0 -layers optimize output.gif
