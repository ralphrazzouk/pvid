from manim import *

class LHCCollision(Scene):
    def construct(self):
        self.camera.background_color = "#141414"

        # Create title
        # title = Text("Proton Bunch Crossing", font_size=36)
        # title.to_edge(UP)
        # self.play(Write(title))
        
        # Define beam parameters
        beam_length = 4
        beam_angle = 25  # degrees
        crossing_angle = beam_angle * DEGREES
        
        # Create beam paths
        beam1_path = Line(
            start=LEFT * beam_length + UP,
            end=RIGHT * beam_length + DOWN,
            color=BLUE_C
        ).rotate(angle=-crossing_angle/2, about_point=ORIGIN)
        
        beam2_path = Line(
            start=RIGHT * beam_length + UP,
            end=LEFT * beam_length + DOWN,
            color=RED_C
        ).rotate(angle=crossing_angle/2, about_point=ORIGIN)
        
        # Create beam labels
        beam1_label = Text("Beam 1", color=BLUE_C, font_size=24)
        beam1_label.next_to(beam1_path.get_start(), UP+LEFT)
        
        beam2_label = Text("Beam 2", color=RED_C, font_size=24)
        beam2_label.next_to(beam2_path.get_start(), UP+RIGHT)
        
        # Draw beam paths
        self.play(
            Create(beam1_path),
            Create(beam2_path),
            Write(beam1_label),
            Write(beam2_label)
        )
        
        # Create proton bunches
        def create_bunch(color):
            bunch = Ellipse(
                width=0.8,
                height=0.3,
                fill_opacity=0.3,
                color=color
            )
            return bunch
        
        # Create bunches for both beams
        bunch1 = create_bunch(BLUE_C)
        bunch2 = create_bunch(RED_C)
        
        # Position bunches at start
        bunch1.move_to(beam1_path.get_start())
        bunch2.move_to(beam2_path.get_start())
        
        # Rotate bunches to align with beams
        bunch1.rotate(angle=-crossing_angle/2)
        bunch2.rotate(angle=crossing_angle/2)
        
        self.play(FadeIn(bunch1), FadeIn(bunch2))
        
        # Animate collision
        collision_time = 3
        self.play(
            bunch1.animate.move_to(ORIGIN),
            bunch2.animate.move_to(ORIGIN),
            run_time=collision_time
        )
        
        # Create collision flash
        flash = Circle(radius=0.3, color=YELLOW)
        flash.move_to(ORIGIN)
        
        self.play(
            flash.animate.scale(3),
            flash.animate.set_opacity(0),
            rate_func=rush_from,
            run_time=0.5
        )
        
        # Create secondary particles
        n_particles = 12
        particles = VGroup()
        for i in range(n_particles):
            angle = i * TAU / n_particles
            particle = Line(
                start=ORIGIN,
                end=UP * 0.8,
                stroke_width=2,
                color=YELLOW
            ).rotate(angle, about_point=ORIGIN)
            particles.add(particle)
        
        # Animate particle spray
        self.play(
            Create(particles),
            particles.animate.scale(2),
            rate_func=linear,
            run_time=0.5
        )
        
        # Final pause
        self.wait(2)







class ThankYou(Scene):
    def construct(self):
        self.camera.background_color = "#141414"

        thankyou = Tex("Thank You!", font_size=64, color=WHITE, )
        self.play(Write(thankyou), run_time=3)

        self.wait(2)