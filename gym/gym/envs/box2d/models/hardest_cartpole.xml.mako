<%
    from rllab.misc.mako_utils import compute_rect_vertices
    cart_width = 2.0 / (12 ** 0.5)
    cart_height = 1.0 / (12 ** 0.5)
    ## cart_width = 0.01
    ## cart_height = 0.01

    plate_width = 0.8
    plate_height = 0.1
    pole_width = 0.07
    pole_height = 0.5
    star_radius = 0.12

    ## pole_width = 0.15
    ## pole_height = 0.8
    anchor_height = 0.1
    noise = opts.get("noise", False)
    if noise:
        import numpy as np
        pole_height += (np.random.rand()-0.5) * pole_height * 1

    cart_friction = 0.01
    plate_friction = 0.5
    ## pole_friction = 0.000002
    pole_friction = 0.5
    goal_pos = -0.2
    goal_size = 0.1
    
    plate_offset = 1.5
    choo = plate_height
%>

<box2d>
  <world timestep="0.03" velitr="10" positr="7">
    <body name="goal" type="static" position="${goal_pos},${cart_height/2}">
      <fixture
              density="1"
              group="-1"
              shape="polygon"
              vertices="${-goal_size}, 0; 0, ${goal_size}; ${goal_size}, 0"
      />
      <fixture
              density="1"
              group="-1"
              shape="polygon"
              vertices="${-goal_size/2}, ${goal_size/2}; ${-goal_size/2}, ${-goal_size}; ${goal_size/2}, ${-goal_size}; ${goal_size/2}, ${goal_size/2}"
      />
    </body>

      ## <body name="pole_anchor" type="dynamic" position="0,${cart_height*2.0 + plate_height/2 + 0/2}">

      ##   <fixture
      ##           density="0"
      ##           friction="${pole_friction}"
      ##           group="-1"
      ##           shape="polygon"
      ##           box="${anchor_height/2},${anchor_height/2}"
      ##   />
      ## </body>
    <body name="pole" type="dynamic" position="0,${plate_offset}">

      ## <fixture
      ##         density="1"
      ##         friction="${pole_friction}"
      ##         group="1"
      ##         shape="polygon"
      ##         vertices="${star_radius/2}, ${-star_radius *0.866 + pole_height/2}; ${star_radius/2}, ${-star_radius * -0.866 +pole_height/2}; ${-star_radius}, ${pole_height/2}"
      ## />
      ## <fixture
      ##         density="1"
      ##         friction="${pole_friction}"
      ##         group="1"
      ##         shape="polygon"
      ##         vertices="${-star_radius/2}, ${star_radius *0.866 + pole_height/2}; ${-star_radius/2}, ${star_radius * -0.866 +pole_height/2}; ${star_radius}, ${pole_height/2}"
      ## />

      <fixture
              density="0.5"
              friction="${pole_friction}"
              group="1"
              shape="polygon"
              vertices="${pole_height/3}, ${pole_height/3}; 0, ${pole_height/3}; 0, ${pole_height/3 + pole_width/1.5}; ${pole_height/3}, ${pole_height/3 + pole_width/1.5}"
      />

      <fixture
              density="0.5"
              friction="${pole_friction}"
              group="1"
              shape="polygon"
              vertices="${-pole_height/3}, ${pole_height/2}; 0, ${pole_height/2}; 0, ${pole_height/2 + pole_width/1.5}; ${-pole_height/3}, ${pole_height/2 + pole_width/1.5}"
      />

      <fixture
              density="0.5"
              friction="${pole_friction}"
              group="1"
              shape="polygon"
              vertices="${-pole_width/2}, 0;${pole_width/2},0; ${pole_width/2}, ${pole_height/1.7}; ${-pole_width/2}, ${pole_height/1.7}"
      />
    </body>

    <body name="plate" type="dynamic" position="0,${plate_offset }">
      <fixture
              density="1"
              friction="${plate_friction}"
              group="-1"
              shape="polygon"
              vertices="${-plate_width/2}, 0;  ${plate_width/2}, 0; ${plate_width/2},  ${-plate_height};  ${-plate_width/2}, ${-plate_height}"
      />

      <fixture
              density="0"
              friction="${plate_friction}"
              group="-1"
              shape="polygon"
              vertices="${-plate_width/3}, 0;  ${plate_width/3}, 0; ${plate_width/3},  ${-plate_height/1.5};  ${-plate_width/3}, ${-plate_height/1.5}"
      />

      <fixture
              density="0.5"
              friction="${plate_friction}"
              group="-1"
              shape="polygon"
              vertices="${plate_width/2}, 0;  ${plate_width/2 - plate_height/2}, 0; ${plate_width/2 - plate_height/2},  ${+pole_height/3};  ${plate_width/2}, ${+pole_height/3}"
      />
      <fixture
              density="0.5"
              friction="${plate_friction}"
              group="-1"
              shape="polygon"
              vertices="${-plate_width/2}, 0;  ${-plate_width/2 + plate_height/2}, 0; ${-plate_width/2 + plate_height/2},  ${+pole_height/3};  ${-plate_width/2}, ${+pole_height/3}"
      />
    </body>
    
    <body name="cart" type="dynamic" position="0,${cart_height/2}">
      <fixture
              density="0.4"
              friction="${cart_friction}"
              group="-1"
              shape="polygon"
              box="${cart_width/2},${cart_height/2}"
      />

    </body>

    <body name="track2" type="static" position="0,${cart_height}">
      <fixture group="-1" shape="polygon" box="100,0.1"/>
    </body>
    <body name="track" type="static" position="0,${cart_height/2}">
      <fixture group="-1" shape="polygon" box="100,0.1"/>
    </body>
    <joint type="revolute" name="plate_joint" bodyA="cart" bodyB="plate" anchor="0,${plate_offset }" limit="-20,20"/>
    <joint type="prismatic" name="track_cart" bodyA="track" bodyB="cart" limit="-1, 1" />
    <state type="xpos" body="goal"/>    ## 0
    <state type="xpos" body="plate"/>   ## 1
    <state type="xvel" body="plate"/>   ## 2
    <state type="apos" body="plate" transform="cos"/>   ## 3   
    <state type="apos" body="plate" transform="cos2"/>   ## 4   
    <state type="apos" body="plate" transform="sin"/>   ## 5  
    <state type="apos" body="plate" transform="sin2"/>   ## 6  
    <state type="avel" body="plate"/>   ## 7
    <state type="xpos" body="pole"/>    ## 8
    <state type="xvel" body="pole"/>    ## 9
    <state type="apos" body="pole" transform="cos"/>    ## 10
    <state type="apos" body="pole" transform="cos2"/>    ## 11
    <state type="apos" body="pole" transform="sin"/>    ## 12
    <state type="apos" body="pole" transform="sin2"/>    ## 13
    <state type="avel" body="pole"/>    ## 14
       
    <control type="force" body="cart" anchor="0,0" direction="1,0" ctrllimit="-1,1"/>
    <control type="torque" joint="plate_joint" ctrllimit="-1,1" />
  </world>
</box2d>

