from superwires import games,color
games.init(screen_width=640,screen_height=480,fps=50)


class Pan(games.Sprite):
      image=games.load_image("C:\\Users\\kaush\\Desktop\\hg.jpg")

      def __init__(self):
            super(Pan,self).__init__(image=Pan.image,x=games.mouse.x,bottom=games.screen.height)
            self.score=games.Text(value=0,size=50,color=color.red,top=5,right=games.screen.width-10)
            games.screen.add(self.score)

      def update(self):
            
                  self.x=games.mouse.x
                  if self.left<0:
                        self.left=0
                  if self.right>games.screen.width:
                        self.right=games.screen.width
                  self.check_catch()
      def check_catch(self):
                  for pizza in self.overlapping_sprites:
                        self.score.value+=10
                        self.score.right=games.screen.width-10
                        pizza.handle_caught()

class Pizza(games.Sprite):
      image=games.load_image("C:\\Users\\kaush\\Desktop\\qw.jpg")
      speed=5
      def __init__(self,x,y=90):
            super(Pizza,self).__init__(image=Pizza.image,x=x,y=y,dy=Pizza.speed)
      def update(self):
            if self.bottom>games.screen.height:
                  self.endgame()
                  self.destroy()
      def handle_caught(self):
            self.destroy()
      def endgame(self):
            end_msg=games.Message(value="Game Over",size=90,color=color.red,x=games.screen.width/2,y=games.screen.height/2,lifetime=5*games.screen.fps,after_death=games.screen.quit)
            games.screen.add(end_msg)


class Chef(games.Sprite):
      image=games.load_image("C:\\Users\\kaush\\Desktop\\images.jpg")

      def __init__(self,y=55,speed=2):
            super(Chef,self).__init__(image=Chef.image,x=games.screen.width/2,y=y,dx=speed)
            self.timedrop=0
      def update(self):
            if self.left<0 or self.right>games.screen.width:
                  self.dx=-self.dx
            self.checkdrop()
      def checkdrop(self):
            if self.timedrop>0:
                  self.timedrop-=1
            else:
                  new_pizza=Pizza(x=self.x)
                  games.screen.add(new_pizza)
                  self.timedrop=int(new_pizza.height*1.3/Pizza.speed)+1

def main():
        wall_img=games.load_image("C:\\Users\\kaush\\Desktop\\gh.jpg",transparent=False)
        games.screen.background=wall_img
        the_chef=Chef()
        games.screen.add(the_chef)
        the_pan=Pan()
        games.screen.add(the_pan)
        games.mouse.is_visible=True
        games.screen.event_grab=True
        games.screen.mainloop()
        
main()
