use bevy::math::bounding::{Aabb2d, BoundingVolume, IntersectsVolume};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CollisionType {
    Left,
    Right,
    Top,
    Bottom,
}

pub fn collide_with_side(ball: Aabb2d, wall: Aabb2d) -> Option<CollisionType> {
    if !ball.intersects(&wall) {
        return None;
    }

    let closest_point = wall.closest_point(ball.center());
    let offset = ball.center() - closest_point;

    let side = if offset.x.abs() > offset.y.abs() {
        if offset.x < 0. {
            CollisionType::Left
        } else {
            CollisionType::Right
        }
    } else {
        if offset.y < 0. {
            CollisionType::Bottom
        } else {
            CollisionType::Top
        }
    };

    Some(side)
}
