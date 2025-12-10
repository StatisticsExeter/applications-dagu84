SELECT lad_ons,
       COUNT(*) AS n_total,
       COUNT(CASE WHEN junction_detail::text = 'Mini-roundabout' THEN 1 END) AS n_miniroundabouts,
       COUNT(CASE WHEN substr(weather_conditions::text, 1, 7) = 'Raining' THEN 1 END) AS n_raining,
       COUNT(CASE WHEN substr(light_conditions::text, 1, 8) = 'Darkness' THEN 1 END) AS n_dark,
       COUNT(CASE WHEN road_surface_conditions::text = 'Dry' THEN 1 END) AS n_dry,
       COUNT(CASE WHEN urban_or_rural_area::text = 'Urban' THEN 1 END) AS n_urban,
       COUNT(CASE WHEN accident_severity::text = 'Slight' THEN 1 END) AS n_slight
FROM dft.stats19_accidents
WHERE substr(accident_index, 1, 4)::int > 2021
GROUP BY lad_ons;
