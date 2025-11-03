use std::ops::Range;

use bevy::{
    input::{
        ButtonState,
        mouse::{AccumulatedMouseScroll, MouseButtonInput, MouseWheel},
    },
    math::bounding::Aabb2d,
    prelude::*,
    window::PrimaryWindow,
};

use crate::collision::{CollisionType, collide_with_side};

mod collision;

fn main() {
    App::new()
        .insert_resource(DragState::default())
        .insert_resource(Camera2dZoomSettings::default())
        // .insert_resource(Score { player: 0, ai: 0 })
        .add_plugins(DefaultPlugins)
        // .add_systems(
        //     Startup,
        //     (
        //         spawn_ball,
        //         spawn_paddles,
        //         spawn_gutters,
        //         spawn_scoreboard,
        //         spawn_camera,
        //     ),
        // )
        // .add_systems(
        //     Update,
        //     (write_system, read_system, conflict_system, conflict2_system),
        // )
        // .add_systems(
        //     FixedUpdate,
        //     (
        //         move_ball.before(project_positions),
        //         project_positions,
        //         handle_collision.after(move_ball),
        //         (handle_player_input, move_ai.after(move_ball)).before(constrain_paddle_position),
        //         constrain_paddle_position,
        //         detect_goal.after(move_ball),
        //         update_scoreboard,
        //     ),
        // )
        // .add_observer(reset_ball)
        // .add_observer(update_score)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_zoom,
                reset_zoom,
                (handle_mouse_input, drag_camera).chain(),
            ),
        )
        .run();
}

fn write_system(mut query: Query<&mut Position>) {
    for mut position in &mut query {
        position.0.x += 0.0;
        position.0.y += 0.0;
    }
}

fn read_system(query: Query<&Position>) {
    for position in &query {
        println!("Position: ({}, {})", position.0.x, position.0.y);
    }
}

fn conflict_system(mut query: Query<&mut Position>) {
    for mut position in &mut query {
        position.0.x += 0.0;
        position.0.y += 0.0;
    }
}

fn conflict1_system(mut query: Query<&mut Position>, mut query2: Query<&mut Position>) {
    for mut position in &mut query {
        position.0.x += 0.0;
        position.0.y += 0.0;
    }
}

fn conflict2_system(
    mut query: Query<&mut Position>,
    mut query2: Query<&mut Position>,
    mut query3: Query<&mut Position>,
) {
    for mut position in &mut query {
        position.0.x += 0.0;
        position.0.y += 0.0;
    }
}

#[derive(Component)]
struct PlayerScore;

#[derive(Component)]
struct AiScore;

fn spawn_scoreboard(mut commands: Commands) {
    let container = Node {
        width: percent(100.0),
        height: percent(100.0),
        justify_content: JustifyContent::Center,
        ..default()
    };

    let header = Node {
        width: px(200.0),
        height: px(100.0),
        ..default()
    };

    let player_score = (
        PlayerScore,
        Text::new("0"),
        TextFont::from_font_size(72.0),
        TextColor(Color::WHITE),
        TextLayout::new_with_justify(Justify::Center),
        Node {
            position_type: PositionType::Absolute,
            top: px(5.0),
            left: px(25.0),
            ..default()
        },
    );

    let ai_score = (
        AiScore,
        Text::new("0"),
        TextFont::from_font_size(72.0),
        TextColor(Color::WHITE),
        TextLayout::new_with_justify(Justify::Center),
        Node {
            position_type: PositionType::Absolute,
            top: px(5.0),
            right: px(25.0),
            ..default()
        },
    );

    commands.spawn((
        container,
        children![(header, children![player_score, ai_score])],
    ));
}

fn update_scoreboard(
    mut player_score: Single<&mut Text, With<PlayerScore>>,
    mut ai_score: Single<&mut Text, (With<AiScore>, Without<PlayerScore>)>,
    score: Res<Score>,
) {
    if score.is_changed() {
        player_score.0 = score.player.to_string();
        ai_score.0 = score.ai.to_string();
    }
}

fn move_ai(
    mut ai: Single<&mut Position, With<Ai>>,
    ball: Single<&Position, (With<Ball>, Without<Ai>)>,
) {
    ai.0.y = ball.0.y;
    // let a_to_b = ball.0 - ai.0;

    // info!(
    //     "position: ball= {:?} ai= {:?} a_to_b.y: {}",
    //     ball.0, ai.0, a_to_b.y
    // );
    // ai.0.y += a_to_b.y.signum() * 5.0;
}

#[derive(Resource)]
struct Score {
    player: u32,
    ai: u32,
}

#[derive(EntityEvent)]
struct Scored {
    #[event_target]
    scorer: Entity,
}

fn detect_goal(
    ball: Single<(&Position, &Collider), With<Ball>>,
    player: Single<Entity, With<Player>>,
    ai: Single<Entity, With<Ai>>,
    window: Single<&Window>,
    mut commands: Commands,
) {
    let (ball_position, ball_collider) = ball.into_inner();
    let half_window_size = window.resolution.size() / 2.0;

    if ball_position.0.x - ball_collider.half_size().x > half_window_size.x {
        commands.trigger(Scored { scorer: *player });
    }

    if ball_position.0.x - ball_collider.half_size().x < -half_window_size.x {
        commands.trigger(Scored { scorer: *ai });
    }
}

fn reset_ball(_event: On<Scored>, ball: Single<(&mut Position, &mut Velocity), With<Ball>>) {
    let (mut ball_position, mut ball_velocity) = ball.into_inner();
    ball_position.0 = Vec2::new(0.0, 0.0);
    ball_velocity.0 = Vec2::new(0.0, 0.0);
}

fn update_score(
    event: On<Scored>,
    mut score: ResMut<Score>,
    ai_q: Query<&Ai>,
    player_q: Query<&Player>,
) {
    if ai_q.get(event.scorer).is_ok() {
        score.ai += 1;
        info!("AI scored! {} - {}", score.player, score.ai);
    }

    if player_q.get(event.scorer).is_ok() {
        score.player += 1;
        info!("Player scored! {} - {}", score.player, score.ai);
    }
}

#[derive(Component)]
#[require(Position, Collider)]
struct Gutter;

const GUTTER_HEIGHT: f32 = 20.0;
const GUTTER_COLOR: Color = Color::srgb(0.0, 0.0, 1.0);

fn spawn_gutters(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    window: Single<&Window>,
) {
    let material = materials.add(GUTTER_COLOR);
    let padding = 20.0;

    let gutter_shape = Rectangle::new(window.resolution.width(), GUTTER_HEIGHT);
    let mesh = meshes.add(gutter_shape);

    let top_gutter_position = Vec2::new(0.0, window.resolution.height() / 2.0 - padding);
    commands.spawn((
        Gutter,
        Mesh2d(mesh.clone()),
        MeshMaterial2d(material.clone()),
        Position(top_gutter_position),
        Collider(gutter_shape),
    ));

    let bottom_gutter_position = Vec2::new(0.0, -window.resolution.height() / 2.0 + padding);
    commands.spawn((
        Gutter,
        Mesh2d(mesh.clone()),
        MeshMaterial2d(material.clone()),
        Position(bottom_gutter_position),
        Collider(gutter_shape),
    ));
}

const PADDLE_SHAPE: Rectangle = Rectangle::new(20.0, 250.0);

#[derive(Component)]
#[require(Position,
    Collider = Collider(PADDLE_SHAPE)
)]
struct Paddle;

const PADDLE_SPEED: f32 = 5.0;

fn handle_player_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut player_paddle_position: Single<&mut Position, With<Player>>,
    mut ball: Single<&mut Velocity, With<Ball>>,
) {
    if keyboard_input.pressed(KeyCode::ArrowUp) {
        player_paddle_position.0.y += PADDLE_SPEED;
    } else if keyboard_input.pressed(KeyCode::ArrowDown) {
        player_paddle_position.0.y += -PADDLE_SPEED;
    } else if keyboard_input.pressed(KeyCode::Space) {
        info!("begin game again!");
        ball.0 = BALL_SPEED;
    }
}

fn constrain_paddle_position(
    paddles: Query<(&mut Position, &Collider), With<Paddle>>,
    gutters: Query<(&Position, &Collider), (With<Gutter>, Without<Paddle>)>,
) {
    for (mut paddle_position, paddle_collider) in paddles {
        for (gutter_position, gutter_collider) in &gutters {
            let paddle_aabb = Aabb2d::new(paddle_position.0, paddle_collider.half_size());
            let gutter_aabb = Aabb2d::new(gutter_position.0, gutter_collider.half_size());

            if let Some(collision) = collide_with_side(paddle_aabb, gutter_aabb) {
                // info!("collision: {collision:?}");
                match collision {
                    CollisionType::Top => {
                        paddle_position.0.y = gutter_position.0.y
                            + gutter_collider.half_size().y
                            + paddle_collider.half_size().y;
                    }
                    CollisionType::Bottom => {
                        paddle_position.0.y = gutter_position.0.y
                            - gutter_collider.half_size().y
                            - paddle_collider.half_size().y;
                    }
                    _ => {}
                }
            }
        }
    }
}

// fn move_paddles(mut paddles: Query<(&mut Position, &Velocity), With<Paddle>>) {
//     for (mut position, velocity) in &mut paddles {
//         position.0 += velocity.0;
//     }
// }

fn spawn_paddles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    window: Single<&Window>,
) {
    let mesh = meshes.add(PADDLE_SHAPE);
    let material = materials.add(Color::srgb(0.0, 1.0, 0.0));
    let half_window_size = window.resolution.size() / 2.0;
    let padding = 20.0;

    let player_position = Vec2::new(-half_window_size.x + padding, 0.0);
    commands.spawn((
        Player,
        Paddle,
        Mesh2d(mesh.clone()),
        MeshMaterial2d(material.clone()),
        Position(player_position),
    ));

    let ai_position = Vec2::new(half_window_size.x - padding, 0.0);
    commands.spawn((
        Ai,
        Paddle,
        Mesh2d(mesh),
        MeshMaterial2d(material),
        Position(ai_position),
    ));
}

fn project_positions(mut positionables: Query<(&mut Transform, &Position)>) {
    for (mut transform, position) in &mut positionables {
        transform.translation = position.0.extend(0.0);
    }
}

#[derive(Component, Default)]
struct Collider(Rectangle);
impl Collider {
    fn half_size(&self) -> Vec2 {
        self.0.half_size
    }
}

fn handle_collision(
    ball: Single<(&mut Velocity, &Position, &Collider), With<Ball>>,
    others: Query<(&Position, &Collider), Without<Ball>>,
) {
    let (mut ball_velocity, ball_position, ball_collider) = ball.into_inner();

    for (other_position, other_collider) in &others {
        let ball_bounds = Aabb2d::new(ball_position.0, ball_collider.half_size());
        let other_bounds = Aabb2d::new(other_position.0, other_collider.half_size());

        if let Some(collision) = collide_with_side(ball_bounds, other_bounds) {
            match collision {
                CollisionType::Left => {
                    ball_velocity.0.x *= -1.0;
                }
                CollisionType::Right => {
                    ball_velocity.0.x *= -1.0;
                }
                CollisionType::Top => {
                    ball_velocity.0.y *= -1.0;
                }
                CollisionType::Bottom => {
                    ball_velocity.0.y *= -1.0;
                }
            }
        }
    }
}

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Ai;

const BALL_SPEED: Vec2 = Vec2::new(2.0, 2.0);

#[derive(Component)]
#[require(Position,
    Velocity = Velocity(BALL_SPEED),
    Collider = Collider(Rectangle::new(10.0, 10.0))
)]
struct Ball;

#[derive(Component)]
struct Velocity(Vec2);

#[derive(Default, Component)]
#[require(Transform)]
struct Position(Vec2);

fn spawn_ball(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mesh = meshes.add(Circle::new(10.0));
    let material = materials.add(Color::srgb(0.0, 1.0, 0.9));
    commands.spawn((Ball, Mesh2d(mesh), MeshMaterial2d(material)));
}

fn move_ball(ball: Single<(&mut Position, &Velocity), With<Ball>>) {
    let (mut position, velocity) = ball.into_inner();
    position.0 += velocity.0;
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((Camera2d, Transform::from_xyz(0.0, 0.0, 0.0)));
}

#[derive(Resource, Default, Debug)]
struct DragState {
    is_dragging: bool,
    last_mouse_pos: Vec2,
    last_camera_pos: Vec3,
}

#[derive(Debug, Resource, Clone)]
struct Camera2dZoomSettings {
    zoom_range: Range<f32>,
    zoom_speed: f32,
    default_scale: f32,
}

impl Default for Camera2dZoomSettings {
    fn default() -> Self {
        Self {
            zoom_range: 0.2..5.0,
            zoom_speed: 0.005,
            default_scale: 1.0,
        }
    }
}

fn handle_mouse_input(
    mut drag_state: ResMut<DragState>,
    mut mouse_button_input: MessageReader<MouseButtonInput>,
    // mouse_button_input: Res<ButtonInput<MouseButton>>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<&Transform, With<Camera2d>>,
) {
    let window = window_query.single().unwrap();
    let camera_transform = camera_query.single().unwrap();

    for event in mouse_button_input.read() {
        if event.button == MouseButton::Left {
            if event.state == ButtonState::Pressed {
                if let Some(mouse_pos) = window.cursor_position() {
                    drag_state.is_dragging = true;
                    drag_state.last_mouse_pos = mouse_pos;
                    drag_state.last_camera_pos = camera_transform.translation;
                    // info!("drag state: {drag_state:?}");
                }
            } else if event.state == ButtonState::Released {
                drag_state.is_dragging = false;
            }
        }
    }
    // if mouse_button_input.just_pressed(MouseButton::Left) {
    //     if let Some(mouse_pos) = window.cursor_position() {
    //         drag_state.is_dragging = true;
    //         drag_state.last_mouse_pos = mouse_pos;
    //         drag_state.last_camera_pos = camera_transform.translation;
    //         info!("drag state: {drag_state:?}");
    //     }
    // } else if mouse_button_input.just_released(MouseButton::Left) {
    //     drag_state.is_dragging = false;
    // }
}

fn drag_camera(
    mut drag_state: ResMut<DragState>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    mut camera_query: Query<&mut Transform, With<Camera2d>>,
) {
    if !drag_state.is_dragging {
        return;
    }

    let window = window_query.single().unwrap();
    let mut camera_transform = camera_query.single_mut().unwrap();

    let Some(current_mouse_pos) = window.cursor_position() else {
        drag_state.is_dragging = false;
        return;
    };

    let mouse_delta = current_mouse_pos - drag_state.last_mouse_pos;
    let camera_delta = Vec3::new(-mouse_delta.x, mouse_delta.y, 0.0);

    camera_transform.translation = drag_state.last_camera_pos + camera_delta;
}

fn handle_zoom(
    mut camera_query: Query<&mut Projection, With<Camera2d>>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    zoom_settings: Res<Camera2dZoomSettings>,
) {
    let mut projection = camera_query.single_mut().unwrap();

    let ortho_proj = match &mut *projection {
        Projection::Orthographic(ortho) => ortho,
        _ => return,
    };

    let scroll_delta = mouse_scroll.delta.y;
    if scroll_delta == 0.0 {
        return;
    }

    let zoom_factor = 1.0 - scroll_delta * zoom_settings.zoom_speed;
    ortho_proj.scale = (ortho_proj.scale * zoom_factor)
        .clamp(zoom_settings.zoom_range.start, zoom_settings.zoom_range.end);
}

fn reset_zoom(
    mut camera_query: Query<(&mut Projection, &mut Transform), With<Camera2d>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    zoom_settings: Res<Camera2dZoomSettings>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) {
        let (mut projection, mut camera_transform) = camera_query.single_mut().unwrap();
        let ortho_proj = match &mut *projection {
            Projection::Orthographic(ortho) => ortho,
            _ => return,
        };

        ortho_proj.scale = zoom_settings.default_scale;
        camera_transform.translation = Vec3::ZERO;
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    zoom_settings: Res<Camera2dZoomSettings>,
) {
    const GRID_DIM_X: usize = 128;
    const GRID_DIM_Y: usize = 128;
    const BLOCK_DIM_X: usize = 32;
    const BLOCK_DIM_Y: usize = 32;

    let m = GRID_DIM_X * BLOCK_DIM_X;
    let n = GRID_DIM_Y * BLOCK_DIM_Y;

    const RECT_SIDE_LENGTH: f32 = 25.0;
    const BLOCK_INTERVAL: f32 = 3.0;
    const THREAD_INTERVAL: f32 = 1.0;

    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scale: zoom_settings.default_scale,
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    let begin_x = -((GRID_DIM_X * BLOCK_DIM_X) as f32 * (RECT_SIDE_LENGTH + BLOCK_INTERVAL)) / 2.0
        + RECT_SIDE_LENGTH;
    let begin_y = (GRID_DIM_Y * BLOCK_DIM_Y) as f32 * (RECT_SIDE_LENGTH + BLOCK_INTERVAL) / 2.0
        - RECT_SIDE_LENGTH;

    for block_x in 0..GRID_DIM_X {
        for block_y in 0..GRID_DIM_Y {
            for thread_x in 0..BLOCK_DIM_X {
                for thread_y in 0..BLOCK_DIM_Y {
                    let idx = block_y * (GRID_DIM_X * BLOCK_DIM_X * BLOCK_DIM_Y)
                        + block_x * (BLOCK_DIM_Y * BLOCK_DIM_X)
                        + thread_y * BLOCK_DIM_X
                        + thread_x;

                    let center_x = begin_x
                        + (block_x * BLOCK_DIM_X + thread_x) as f32
                            * (THREAD_INTERVAL + RECT_SIDE_LENGTH)
                        + (BLOCK_INTERVAL - THREAD_INTERVAL) * block_x as f32;

                    let center_y = begin_y
                        - ((block_y * BLOCK_DIM_Y + thread_y) as f32
                            * (THREAD_INTERVAL + RECT_SIDE_LENGTH)
                            + (BLOCK_INTERVAL - THREAD_INTERVAL) * block_y as f32);
                    // info!("center: x= {center_x} y= {center_y}");

                    commands.spawn((
                        Mesh2d(meshes.add(Rectangle::new(RECT_SIDE_LENGTH, RECT_SIDE_LENGTH))),
                        MeshMaterial2d(materials.add(Color::srgba(0.2, 0.4, 0.8, 0.9))),
                        Transform::from_xyz(center_x, center_y, 0.0),
                    ));

                    commands.spawn((
                        Text2d::new(format!("{idx}")),
                        TextFont {
                            font_size: 5.0,
                            ..default()
                        },
                        Transform::from_xyz(center_x, center_y + 7.0, 1.0),
                    ));

                    // 2d thoughts
                    let x = block_y * BLOCK_DIM_Y + thread_y;
                    let y = block_x * BLOCK_DIM_X + thread_x;
                    if x < m && y < n {
                        commands.spawn((
                            Text2d::new(format!("{x} {y}")),
                            TextFont {
                                font_size: 4.0,
                                ..default()
                            },
                            Transform::from_xyz(center_x, center_y, 1.0),
                        ));
                    }

                    // 1d thoughts
                    if idx < m * n {
                        let x = idx / m;
                        let y = idx % m;
                        commands.spawn((
                            Text2d::new(format!("{x} {y}")),
                            TextFont {
                                font_size: 4.0,
                                ..default()
                            },
                            Transform::from_xyz(center_x, center_y - 7.0, 1.0),
                        ));
                    }
                }
            }
        }
    }

    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: 1.0,
        ..default()
    });
}
