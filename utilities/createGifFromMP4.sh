ffmpeg -i movie.mp4  -vf "fps=10,scale=1000:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 10 - -loop 0 -layers optimize output.gif
