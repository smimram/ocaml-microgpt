open Autograd

(* Let's try to learn f(x) = x². *)
let () =
  Random.self_init ();

  (* Generate dataset. *)
  let dataset = List.map (fun x -> x, x *. x) [-1.0; -0.8; -0.6; -0.4; -0.2; 0.0; 0.2; 0.4; 0.6; 0.8; 1.0] in

  (* Train a network with one hidden layer of size 6. *)
  let layer1 = Matrix.init 6 1 (fun _ _ -> const (Random.float 1.)) in
  let layer2 = Matrix.init 1 6 (fun _ _ -> const (Random.float 1.)) in
  let params = List.flatten @@ List.map Matrix.coefficients [layer1; layer2] in

  Printf.printf "num params: %d\n" (List.length params);

  let net x = x |> Matrix.ap layer1 |> Matrix.ap layer2 in

  let learning_rate = 0.1 in
  let steps = 1_000 in
  for step = 0 to steps - 1 do
    let loss = ref @@ const 0. in
    List.iter (fun (x,y) ->
        let y' = net [|const x|] in
        loss := add !loss (powc (sub y'.(0) (const y)) 2.)
      ) dataset;
    let loss = !loss in
    backward loss;

    Printf.printf "step: %4d | loss: %.4f\n%!" step (value loss);

    let lr = learning_rate *. (1. -. float step /. float steps) in
    List.iter (fun p ->
        p.value <- value p -. lr *. grad p;
        p.grad <- 0.
      ) params;
  done;

  (* Profit. *)
  let xs = [-1.0; -0.5; 0.0; 0.1; 0.5; 1.] in
  List.iter
    (fun x ->
      let y = value (net [|const x|]).(0) in
      Printf.printf "f(%f) = %f instead of %f\n" x y (x *. x)
    ) xs

